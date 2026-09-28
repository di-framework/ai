export type AllowedDirectories = readonly string[] | (() => readonly string[]);

export function resolveAllowedDirectories(dirs: AllowedDirectories): readonly string[] {
  return typeof dirs === 'function' ? dirs() : dirs;
}
