import { Composition } from "remotion";
import { DevDuckIntro } from "./DevDuckIntro";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="DevDuckIntro"
        component={DevDuckIntro}
        durationInFrames={30 * 90} // 90 seconds at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
