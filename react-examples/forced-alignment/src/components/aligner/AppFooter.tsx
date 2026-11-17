import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { appZipUrl, currentModel, docUrl } from "@/lib/model-config";
import type { VersionInfo } from "@/lib/sherpa/engine";
import { useState } from "react";

const githubUrl = "https://github.com/k2-fsa/sherpa-onnx";
const sourceUrl =
  "https://github.com/k2-fsa/sherpa-onnx/tree/master/react-examples/forced-alignment";

export function AppFooter({ version }: { version: VersionInfo | null }) {
  const model = currentModel();
  const zipUrl = appZipUrl(model);
  const zipName = `sherpa-onnx-react-fa-${model.dir}.zip`;
  const zipDir = zipName.replace(/\.zip$/, "");
  const [copied, setCopied] = useState(false);
  const commands =
    `wget ${zipUrl}
unzip ${zipName}
cd ${zipDir}
python3 -m http.server 8000`;
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
            To run it locally:
          </p>
          <div className="relative">
            <button
              type="button"
              className="absolute right-1.5 top-1.5 rounded-md border bg-white px-2.5 py-0.5 text-xs"
              onClick={() => {
                navigator.clipboard
                  .writeText(commands)
                  .then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1500);
                  })
                  .catch(() => {});
              }}
            >
              {copied ? "Copied!" : "Copy"}
            </button>
            <pre
              className="text-muted-foreground overflow-x-auto whitespace-pre rounded-md border p-2 font-mono"
              style={{ fontSize: "11px" }}
            >
              {commands}
            </pre>
          </div>
          <p className="text-muted-foreground">
            then open a browser and visit{" "}
            <a className="underline" href="http://localhost:8000">
              http://localhost:8000
            </a>
            . See the documentation for details.
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
