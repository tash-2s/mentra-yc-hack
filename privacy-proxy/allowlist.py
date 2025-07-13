from pathlib import Path
import numpy as np, face_recognition, logging
import threading
import asyncio
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

log = logging.getLogger("allowlist")

class AllowListManager:
    def __init__(self, folder:Path, distance:float, consented_captures_dir: Optional[Path] = None):
        self.folder = Path(folder)
        self.distance = distance
        self.consented_captures_dir = consented_captures_dir
        
        # Thread safety for dynamic updates
        self._lock = threading.RLock()
        
        # Load initial allowlist
        self._encodings, self._names = self._load()
        
        # Load any existing consented captures
        if self.consented_captures_dir and self.consented_captures_dir.exists():
            self._load_consented_captures()
        
        # Start monitoring for new consented captures
        self._observer = None
        if self.consented_captures_dir:
            self._start_monitoring()

    # --- public --------------------------------------------------
    def match(self, face_img: np.ndarray) -> str|None:
        # Since face_img is already a cropped face, we need to tell face_recognition
        # that the face location is the entire image
        h, w = face_img.shape[:2]
        # face_locations format is (top, right, bottom, left)
        face_location = [(0, w, h, 0)]
        
        try:
            # Use the full image as the face location since it's pre-cropped
            enc = face_recognition.face_encodings(face_img, known_face_locations=face_location, num_jitters=1)
            if not enc:  # no encodings generated
                return None
            enc = enc[0]
            
            # Thread-safe access to encodings
            with self._lock:
                if self._encodings.size == 0:
                    return None
                    
                dists = face_recognition.face_distance(self._encodings, enc)
                idx = np.argmin(dists)
                return self._names[idx] if dists[idx] < self.distance else None
        except Exception as e:
            log.warning(f"Failed to encode face: {e}")
            return None
    
    def stop(self):
        """Stop monitoring for new faces"""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            log.info("Stopped monitoring for consented captures")

    # --- private -------------------------------------------------
    def _load(self):
        encodings, names = [], []
        if not self.folder.exists():
            log.warning(f"Allow-list directory '{self.folder}' does not exist")
            log.warning("All faces will be blurred")
            return np.empty((0,128)), names

        for person_dir in filter(Path.is_dir, self.folder.iterdir()):
            for img_path in person_dir.glob("*.[jp][pn]g"):
                try:
                    img = face_recognition.load_image_file(img_path)
                    face_locations = face_recognition.face_locations(img, model="hog")
                    if not face_locations:
                        log.warning(f"No face found in {img_path}")
                        continue
                    enc = face_recognition.face_encodings(img, face_locations)[0]
                    encodings.append(enc)
                    names.append(person_dir.name)
                    log.info(f"Loaded face encoding for {person_dir.name} from {img_path.name}")
                except Exception as e:
                    log.error(f"Failed to load {img_path}: {e}")

        if encodings:
            log.info(f"Loaded {len(encodings)} face encodings "
                     f"for {len(set(names))} people")
        else:
            log.warning("No allow-list faces found; everyone will be blurred")
        return np.vstack(encodings) if encodings else np.empty((0,128)), names
    
    def _load_consented_captures(self):
        """Load existing faces from consented captures directory"""
        if not self.consented_captures_dir or not self.consented_captures_dir.exists():
            return
            
        count = 0
        for img_path in self.consented_captures_dir.glob("*.jpg"):
            if self._load_single_consent_face(img_path):
                count += 1
                
        if count > 0:
            log.info(f"Loaded {count} faces from consented captures")
    
    def _load_single_consent_face(self, img_path: Path) -> bool:
        """Load a single consented face and add to allowlist"""
        try:
            # Parse filename: YYYY-MM-DD-HH-MM-SS_PersonName.jpg
            filename = img_path.stem
            parts = filename.split('_', 1)
            if len(parts) < 2:
                log.warning(f"Invalid consent filename format: {img_path.name}")
                return False
                
            person_name = parts[1] if parts[1] != "anonymous" else f"Anonymous_{parts[0]}"
            
            # Load and encode the face
            img = face_recognition.load_image_file(img_path)
            
            # Detect face in the image (same as allowlist loading)
            face_locations = face_recognition.face_locations(img, model="hog")
            if not face_locations:
                log.warning(f"No face found in {img_path}")
                return False
            
            encodings = face_recognition.face_encodings(img, face_locations)
            if not encodings:
                log.warning(f"No face encoding generated for {img_path}")
                return False
                
            # Add to allowlist with thread safety
            with self._lock:
                if self._encodings.size == 0:
                    self._encodings = np.array([encodings[0]])
                else:
                    self._encodings = np.vstack([self._encodings, encodings[0]])
                self._names.append(person_name)
                
            log.info(f"Added consented face to allowlist: {person_name} from {img_path.name}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load consented face {img_path}: {e}")
            return False
    
    def _start_monitoring(self):
        """Start watchdog observer to monitor for new consented faces"""
        class ConsentHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
                
            def on_created(self, event):
                src_path = str(event.src_path)
                if not event.is_directory and src_path.endswith('.jpg'):
                    # Process new consent face after a short delay to ensure file is written
                    threading.Timer(0.5, self.manager._process_new_consent, args=[src_path]).start()
        
        self._observer = Observer()
        self._observer.schedule(ConsentHandler(self), str(self.consented_captures_dir), recursive=False)
        self._observer.start()
        log.info(f"Started monitoring for new consented faces in: {self.consented_captures_dir}")
    
    def _process_new_consent(self, file_path: str):
        """Process a newly created consent face file"""
        img_path = Path(file_path)
        if img_path.exists():
            if self._load_single_consent_face(img_path):
                log.info(f"Successfully added new consented face from: {img_path.name}")
            else:
                log.warning(f"Failed to add new consented face from: {img_path.name}")
