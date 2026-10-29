export class SessionRegistry {
  constructor() { this.sessions = new Map(); }
  registerSession(id, owner) { this.sessions.set(id, owner); }
  resolveSession(id) { return this.sessions.get(id); }
}
