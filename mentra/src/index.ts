import { AppServer, AppSession, StreamType } from '@mentra/sdk';

const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => { throw new Error('PACKAGE_NAME is not set in .env file'); })();
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY ?? (() => { throw new Error('MENTRAOS_API_KEY is not set in .env file'); })();
const PORT = parseInt(process.env.PORT || '3000');

const RTMP_URL = 'rtmp://localhost:1945/live/test'; // FIXME

class MyApp extends AppServer {
  protected async onSession(session: AppSession, userId: string): Promise<void> {
    session.logger.info(`New session: ${userId}`);

    session.events.onButtonPress(async (button) => {
      this.logger.info(`Button pressed: ${button.buttonId}, type: ${button.pressType}`);
      this.logger.info(`isStreaming: ${session.camera.isCurrentlyStreaming()}`);

      if (session.camera.isCurrentlyStreaming()) {
        await session.camera.startStream({
          rtmpUrl: RTMP_URL,
        });
      } else {
        await session.camera.stopStream();
      }
    });

    // Subscribe to RTMP stream status updates
    const statusUnsubscribe = session.camera.onStreamStatus((status) => {
      console.log(`Stream status: ${status.status}`);

      if (status.status === 'active') {
        console.log('🟢 RTMP stream is live!');
        session.layouts.showTextWall('🟢 Stream is live!');

        if (status.stats) {
          console.log(`Stats:
            Bitrate: ${status.stats.bitrate} bps
            FPS: ${status.stats.fps}
            Duration: ${status.stats.duration}s
            Dropped Frames: ${status.stats.droppedFrames}
          `);
        }
      } else if (status.status === 'error') {
        console.error(`❌ Stream error: ${status.errorDetails}`);
        session.layouts.showTextWall(`❌ Stream Error\n\n${status.errorDetails || 'Unknown error'}`);
      } else if (status.status === 'initializing') {
        console.log('📡 Initializing RTMP connection...');
        session.layouts.showTextWall('📡 Connecting to RTMP server...');
      } else if (status.status === 'connecting') {
        console.log('🔗 Connecting to RTMP server...');
        session.layouts.showTextWall('🔗 Establishing connection...');
      } else if (status.status === 'stopped') {
        console.log('🔴 Stream stopped');
        session.layouts.showTextWall('🔴 Stream stopped');
      }
    });

    const healthCheckInterval = setInterval(() => {
      if (session.camera.isCurrentlyStreaming()) {
        const status = session.camera.getStreamStatus();
        if (status?.stats) {
          // Alert if dropped frames are high
          const dropRate = (status.stats.droppedFrames / (status.stats.fps * status.stats.duration)) * 100;
          if (dropRate > 5) { // More than 5% dropped frames
            console.warn(`⚠️ High drop rate: ${dropRate.toFixed(1)}%`);
            session.layouts.showTextWall(`⚠️ Poor connection\n\nDropped frames: ${dropRate.toFixed(1)}%`);
          }
        }
      }
    }, 10000);

    this.addCleanupHandler(statusUnsubscribe);
    this.addCleanupHandler(() => clearInterval(healthCheckInterval));
  }
}

const app = new MyApp({
  packageName: PACKAGE_NAME,
  apiKey: MENTRAOS_API_KEY,
  port: PORT
});

app.start().catch(console.error);
