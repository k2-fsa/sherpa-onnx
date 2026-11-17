import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { appZipUrl, currentModel, docUrl } from "@/lib/model-config";
import type { VersionInfo } from "@/lib/sherpa/engine";

const githubUrl = "https://github.com/k2-fsa/sherpa-onnx";
const sourceUrl =
  "https://github.com/k2-fsa/sherpa-onnx/tree/master/react-examples/forced-alignment";

export function AppFooter({ version }: { version: VersionInfo | null }) {
  const model = currentModel();
  return (
    <footer className="mt-8 pb-10 space-y-3">
      {version && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Version</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-1">
            <p>sherpa-onnx: {version.version}</p>
            <p>Git SHA1: {version.gitSha1}</p>
            <p>Git date: {version.gitDate}</p>
            <p>onnxruntime: {version.onnxruntime}</p>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Current model</CardTitle>
        </CardHeader>
        <CardContent className="text-sm space-y-1">
          <p className="font-medium">{model.name}</p>
          <p className="text-muted-foreground break-all">
            <a className="underline" href={model.url}>
              Download the model
            </a>
            {" · "}
            <a className="underline" href={appZipUrl(model)}>
              Download this app (.zip)
            </a>
            {" · "}
            <a className="underline" href={docUrl}>
              Documentation
            </a>
            {" · "}
            <a className="underline" href={sourceUrl}>
              Source code
            </a>
          </p>
          <p className="text-muted-foreground">
            Runs fully in your browser — offline-capable, no audio is uploaded anywhere.
          </p>
          <p className="text-muted-foreground">
            To run it locally: download the app zip above, unpack it, and open
            the folder with any static HTTP server (for example
            <code> python3 -m http.server 8000</code>), then visit the
            printed address in your browser. See the documentation for
            details.
          </p>
        </CardContent>
      </Card>
      <p className="text-center text-xs text-muted-foreground">
        Powered by{" "}
        <a className="underline" href={githubUrl}>
          sherpa-onnx
        </a>
      </p>
    </footer>
  );
}
