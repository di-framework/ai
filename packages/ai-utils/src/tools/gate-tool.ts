import type { ToolCallback } from '@di-framework/ai';
import type { SkillsRuntime } from '../skills/skills-runtime.ts';

export function gateToolCallback(tool: ToolCallback, runtime: SkillsRuntime): ToolCallback {
  return {
    toolDefinition: tool.toolDefinition,
    toolMetadata: tool.toolMetadata,
    call(toolInput: string, toolContext) {
      const name = tool.toolDefinition.name;
      if (!runtime.isToolAllowed(name)) {
        return runtime.deniedToolMessage(name);
      }
      return tool.call(toolInput, toolContext);
    },
  };
}
