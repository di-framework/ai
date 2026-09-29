import { toolCallbackAsMcpTool } from '@di-framework/ai';
import { createSkillsToolbox, type SkillsToolboxOptions } from './skills-toolbox.ts';

/**
 * Expose a skills toolbox as MCP descriptors + handlers.
 */
export function skillsToolboxAsMcp(options: SkillsToolboxOptions = {}) {
  return createSkillsToolbox(options).tools.map((tool) => toolCallbackAsMcpTool(tool));
}
