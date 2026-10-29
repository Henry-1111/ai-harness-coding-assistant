import { SessionRegistry } from './session.js';
export function recordSessionAudit(id, registry) {
  if (!(registry instanceof SessionRegistry)) throw new Error('invalid registry');
  return { id, recorded: true };
}
