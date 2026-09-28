import { describe, expect, test } from 'bun:test';
import type { AuthorizationManager, Principal } from '@di-framework/auth';
import { createPrincipal } from '@di-framework/auth';
import {
  createToolCallingManager,
  functionToolCallback,
  isToolResponseMessage,
  Prompt,
  Tool,
  type ToolAuthorizationContext,
  ToolSet,
  toolAuthorizationAdvisor,
  toolCall,
  toolCallbacksFromBean,
  toolCallResponse,
  userMessage,
} from '../src/index.ts';
import type { ToolExecutionAdvisor } from '../src/model/tool/tool-execution-advisor.ts';

describe('Tool Execution Authorization & Interception', () => {
  const alicePrincipal = createPrincipal({
    sub: 'user-alice',
    method: 'bearer',
  });

  test('allowed execution invokes callback once and returns result', async () => {
    let callCount = 0;
    const callback = functionToolCallback({
      name: 'getWeather',
      description: 'Get weather for city',
      call: ({ city }: { city: string }) => {
        callCount++;
        return `Weather for ${city} is 72F`;
      },
    });

    let authorizedPrincipal: Principal | undefined;
    let authorizedContext: ToolAuthorizationContext | undefined;

    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize(principal, context) {
        authorizedPrincipal = principal;
        authorizedContext = context;
        return { allowed: true };
      },
    };

    const manager = createToolCallingManager({
      advisors: [toolAuthorizationAdvisor({ authorizationManager: mockManager })],
    });

    const prompt = new Prompt([userMessage('What is the weather in Boston?')], {
      toolCallbacks: [callback],
      toolContext: { principal: alicePrincipal },
    });

    const mockResponse = toolCallResponse([toolCall('call-1', 'getWeather', { city: 'Boston' })]);

    const result = await manager.executeToolCalls(prompt, mockResponse);

    expect(callCount).toBe(1);
    expect(authorizedPrincipal).toEqual(alicePrincipal);
    expect(authorizedContext).toBeDefined();
    expect(authorizedContext?.transport).toBe('ai-tool');
    expect(authorizedContext?.tool).toBe('getWeather');
    expect(authorizedContext?.arguments).toEqual({ city: 'Boston' });

    const lastMsg = result.conversationHistory[result.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsg!)).toBe(true);
    if (isToolResponseMessage(lastMsg!)) {
      expect(lastMsg.responses[0]?.responseData).toContain('Weather for Boston is 72F');
    }
  });

  test('denied execution never calls callback and returns generic failure message without leaking details', async () => {
    let callCount = 0;
    const callback = functionToolCallback({
      name: 'deleteDatabase',
      description: 'Delete entire database',
      call: () => {
        callCount++;
        return 'Database deleted';
      },
    });

    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize() {
        return {
          allowed: false,
          reason: 'SUPER_SECRET_INTERNAL_POLICY_DENIED: Alice lacks DB_ADMIN_ROLE',
          detail: { internalRuleId: 'RULE-999' },
        };
      },
    };

    const manager = createToolCallingManager({
      authorizationManager: mockManager,
    });

    const prompt = new Prompt([userMessage('Delete everything')], {
      toolCallbacks: [callback],
      toolContext: { principal: alicePrincipal },
    });

    const mockResponse = toolCallResponse([toolCall('call-1', 'deleteDatabase', {})]);

    const result = await manager.executeToolCalls(prompt, mockResponse);

    expect(callCount).toBe(0);
    const lastMsg = result.conversationHistory[result.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsg!)).toBe(true);
    if (isToolResponseMessage(lastMsg!)) {
      expect(lastMsg.responses[0]?.responseData).toBe('Tool execution unauthorized');
      expect(lastMsg.responses[0]?.responseData).not.toContain('SUPER_SECRET');
      expect(lastMsg.responses[0]?.responseData).not.toContain('DB_ADMIN_ROLE');
      expect(lastMsg.responses[0]?.responseData).not.toContain('RULE-999');
    }
  });

  test('missing principal or authorization manager failure fails closed', async () => {
    let callCount = 0;
    const callback = functionToolCallback({
      name: 'sendEmail',
      call: () => {
        callCount++;
        return 'Email sent';
      },
    });

    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize() {
        return { allowed: true };
      },
    };

    const advisor = toolAuthorizationAdvisor({ authorizationManager: mockManager });
    const manager = createToolCallingManager({ advisors: [advisor] });

    const mockResponse = toolCallResponse([toolCall('call-1', 'sendEmail', {})]);

    // Scenario A: Missing principal in ToolContext
    const promptNoPrincipal = new Prompt([userMessage('Send email')], {
      toolCallbacks: [callback],
      toolContext: {}, // empty context
    });

    const resultNoPrincipal = await manager.executeToolCalls(promptNoPrincipal, mockResponse);
    expect(callCount).toBe(0);
    const lastMsgNoPrincipal =
      resultNoPrincipal.conversationHistory[resultNoPrincipal.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsgNoPrincipal!)).toBe(true);
    if (isToolResponseMessage(lastMsgNoPrincipal!)) {
      expect(lastMsgNoPrincipal.responses[0]?.responseData).toBe('Tool execution unauthorized');
    }

    // Scenario B: AuthorizationManager throws an error
    const throwingManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize() {
        throw new Error('Database connection failed in AuthorizationManager');
      },
    };

    const throwingManagerInstance = createToolCallingManager({
      advisors: [toolAuthorizationAdvisor({ authorizationManager: throwingManager })],
    });

    const promptWithPrincipal = new Prompt([userMessage('Send email')], {
      toolCallbacks: [callback],
      toolContext: { principal: alicePrincipal },
    });

    const resultThrowing = await throwingManagerInstance.executeToolCalls(
      promptWithPrincipal,
      mockResponse,
    );
    expect(callCount).toBe(0);
    const lastMsgThrowing =
      resultThrowing.conversationHistory[resultThrowing.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsgThrowing!)).toBe(true);
    if (isToolResponseMessage(lastMsgThrowing!)) {
      expect(lastMsgThrowing.responses[0]?.responseData).toBe('Tool execution unauthorized');
    }

    // Scenario C: Invalid/unresolvable manager
    const unresolvableAdvisor = toolAuthorizationAdvisor({
      managerToken: 'NON_EXISTENT_MANAGER_TOKEN',
      container: {
        has: () => false,
        resolve: () => undefined,
      } as unknown as import('@di-framework/auth').AuthContainer,
    });
    const unresolvableManager = createToolCallingManager({ advisors: [unresolvableAdvisor] });

    const resultUnresolvable = await unresolvableManager.executeToolCalls(
      promptWithPrincipal,
      mockResponse,
    );
    expect(callCount).toBe(0);
    const lastMsgUnresolvable =
      resultUnresolvable.conversationHistory[resultUnresolvable.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsgUnresolvable!)).toBe(true);
    if (isToolResponseMessage(lastMsgUnresolvable!)) {
      expect(lastMsgUnresolvable.responses[0]?.responseData).toBe('Tool execution unauthorized');
    }
  });

  test('model arguments cannot overwrite trusted principal in context', async () => {
    let checkedPrincipal: Principal | undefined;
    let checkedArgs: unknown;

    const callback = functionToolCallback({
      name: 'transferMoney',
      call: () => 'Transferred $1000',
    });

    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize(principal, context) {
        checkedPrincipal = principal;
        checkedArgs = context.arguments;
        return { allowed: true };
      },
    };

    const manager = createToolCallingManager({
      authorizationManager: mockManager,
    });

    // Model tries to inject a spoofed principal in arguments!
    const spoofedArgs = JSON.stringify({
      amount: 1000,
      principal: {
        sub: 'root-admin',
        method: 'password',
      },
      user: {
        sub: 'super-admin',
      },
    });

    const prompt = new Prompt([userMessage('Transfer $1000')], {
      toolCallbacks: [callback],
      toolContext: { principal: alicePrincipal },
    });

    const mockResponse = toolCallResponse([toolCall('call-1', 'transferMoney', spoofedArgs)]);

    await manager.executeToolCalls(prompt, mockResponse);

    // Trusted principal MUST be alicePrincipal, NOT spoofed root-admin
    expect(checkedPrincipal).toBeDefined();
    expect(checkedPrincipal?.sub).toBe('user-alice');
    expect(checkedPrincipal).toEqual(alicePrincipal);

    // Context arguments carries the parsed model arguments
    expect(checkedArgs).toEqual({
      amount: 1000,
      principal: {
        sub: 'root-admin',
        method: 'password',
      },
      user: {
        sub: 'super-admin',
      },
    });
  });

  test('carries opaque metadata for @Tool methods, @ToolSet beans, and function callbacks', async () => {
    const capturedMetadata: unknown[] = [];

    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize(_principal, context) {
        capturedMetadata.push(context.metadata);
        return { allowed: true };
      },
    };

    const manager = createToolCallingManager({
      authorizationManager: mockManager,
    });

    // 1. Function callback with auth metadata
    const fnCallback = functionToolCallback({
      name: 'fnTool',
      auth: { role: 'OPERATOR' },
      call: () => 'fnResult',
    });

    // 2. Class decorated with @ToolSet and @Tool
    @ToolSet({ auth: { scope: 'user-management' } })
    class AdminTools {
      @Tool({ auth: { permission: 'user:delete' } })
      deleteUser() {
        return 'user deleted';
      }

      @Tool()
      listUsers() {
        return 'user list';
      }
    }

    const adminToolsInstance = new AdminTools();
    const beanCallbacks = toolCallbacksFromBean(adminToolsInstance);

    const allCallbacks = [fnCallback, ...beanCallbacks];

    const prompt = new Prompt([userMessage('Run tools')], {
      toolCallbacks: allCallbacks,
      toolContext: { principal: alicePrincipal },
    });

    // Executing fnTool
    await manager.executeToolCalls(prompt, toolCallResponse([toolCall('1', 'fnTool', {})]));

    // Executing deleteUser (merged @ToolSet + @Tool auth)
    await manager.executeToolCalls(prompt, toolCallResponse([toolCall('2', 'deleteUser', {})]));

    // Executing listUsers (only @ToolSet auth)
    await manager.executeToolCalls(prompt, toolCallResponse([toolCall('3', 'listUsers', {})]));

    expect(capturedMetadata).toHaveLength(3);
    expect(capturedMetadata[0]).toEqual({ role: 'OPERATOR' });
    expect(capturedMetadata[1]).toEqual({
      scope: 'user-management',
      permission: 'user:delete',
    });
    expect(capturedMetadata[2]).toEqual({ scope: 'user-management' });
  });

  test('authorizes multiple tool calls in a single model turn independently', async () => {
    const executed: string[] = [];

    const readTool = functionToolCallback({
      name: 'readData',
      call: () => {
        executed.push('readData');
        return 'data content';
      },
    });

    const writeTool = functionToolCallback({
      name: 'writeData',
      call: () => {
        executed.push('writeData');
        return 'data written';
      },
    });

    // Manager allows readData, denies writeData
    const mockManager: AuthorizationManager<ToolAuthorizationContext> = {
      authorize(_principal, context) {
        if (context.tool === 'readData') {
          return { allowed: true };
        }
        return { allowed: false, reason: 'Write access denied' };
      },
    };

    const manager = createToolCallingManager({
      authorizationManager: mockManager,
    });

    const prompt = new Prompt([userMessage('Read and write data')], {
      toolCallbacks: [readTool, writeTool],
      toolContext: { principal: alicePrincipal },
    });

    const multiCallResponse = toolCallResponse([
      toolCall('c1', 'readData', {}),
      toolCall('c2', 'writeData', {}),
    ]);

    const result = await manager.executeToolCalls(prompt, multiCallResponse);

    // Only readData was executed
    expect(executed).toEqual(['readData']);

    const lastMsg = result.conversationHistory[result.conversationHistory.length - 1];
    expect(isToolResponseMessage(lastMsg!)).toBe(true);
    if (isToolResponseMessage(lastMsg!)) {
      expect(lastMsg.responses).toHaveLength(2);
      expect(lastMsg.responses[0]?.responseData).toBe('data content');
      expect(lastMsg.responses[1]?.responseData).toBe('Tool execution unauthorized');
    }
  });

  test('executes custom tool execution advisors in order', async () => {
    const trace: string[] = [];

    const advisor1: ToolExecutionAdvisor = {
      name: 'First Advisor',
      order: 10,
      async adviseExecution(ctx, next) {
        trace.push('advisor1-before');
        const res = await next(ctx);
        trace.push('advisor1-after');
        return res;
      },
    };

    const advisor2: ToolExecutionAdvisor = {
      name: 'Second Advisor',
      order: 20,
      async adviseExecution(ctx, next) {
        trace.push('advisor2-before');
        const res = await next(ctx);
        trace.push('advisor2-after');
        return res;
      },
    };

    const callback = functionToolCallback({
      name: 'testTool',
      call: () => {
        trace.push('tool-call');
        return 'ok';
      },
    });

    const manager = createToolCallingManager({
      advisors: [advisor2, advisor1], // Passed out of order to verify sorting by order
    });

    const prompt = new Prompt([userMessage('Test')], {
      toolCallbacks: [callback],
    });

    await manager.executeToolCalls(prompt, toolCallResponse([toolCall('1', 'testTool', {})]));

    expect(trace).toEqual([
      'advisor1-before',
      'advisor2-before',
      'tool-call',
      'advisor2-after',
      'advisor1-after',
    ]);
  });
});
