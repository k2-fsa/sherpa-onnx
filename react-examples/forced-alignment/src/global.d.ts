export {};

declare global {
  interface Window {
    Module?: Record<string, unknown>;
    ForcedAligner?: new (
      config: Record<string, unknown>,
      module: unknown,
    ) => unknown;
    OfflineStream?: new (handle: number, module: unknown) => unknown;
  }
}
