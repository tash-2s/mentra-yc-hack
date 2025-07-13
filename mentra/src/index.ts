import { AppServer, AppSession, StreamType } from '@mentra/sdk';

const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => { throw new Error('PACKAGE_NAME is not set in .env file'); })();
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY ?? (() => { throw new Error('MENTRAOS_API_KEY is not set in .env file'); })();
const PORT = parseInt(process.env.PORT || '3000');

const RTMP_URL = 'rtmp://4.tcp.us-cal-1.ngrok.io:11976/live/test'; // FIXME

class MyApp extends AppServer {
  protected async onSession(session: AppSession, userId: string): Promise<void> {
    session.logger.info(`New session: ${userId}`);

    session.events.onButtonPress(async (button) => {
      this.logger.info(`Button pressed: ${button.buttonId}, type: ${button.pressType}`);

      if (session.camera.isCurrentlyStreaming()) {
        await session.camera.stopStream();
      } else {
        await session.camera.startStream({
          rtmpUrl: RTMP_URL,
        });
      }
    });

    // Subscribe to RTMP stream status updates
    session.subscribe(StreamType.RTMP_STREAM_STATUS);
    const statusUnsubscribe = session.camera.onStreamStatus((status) => {
      console.log(`Stream status: ${JSON.stringify(status)}`);
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
