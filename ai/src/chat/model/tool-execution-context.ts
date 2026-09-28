import type { ToolCallingManager } from '../../model/tool/tool-calling-manager.ts';
import type { Prompt } from '../prompt/prompt.ts';
import type { ChatModel } from './chat-model.ts';
import type { ChatResponse } from './chat-response.ts';

/** Internal host-only context; never included in model options or serialized prompts. */
export const TOOL_MANAGER_CONTEXT = 'di-framework.internal.toolCallingManager';
export const CALL_WITH_TOOL_MANAGER = Symbol('callWithToolManager');

export interface ToolExecutingChatModel extends ChatModel {
  [CALL_WITH_TOOL_MANAGER](prompt: Prompt, manager?: ToolCallingManager): Promise<ChatResponse>;
}

export function isToolExecutingChatModel(model: ChatModel): model is ToolExecutingChatModel {
  return CALL_WITH_TOOL_MANAGER in model;
}
