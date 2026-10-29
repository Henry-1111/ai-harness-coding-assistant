export interface QuotaSnapshot { remaining: number; }
export class QuotaLedger {
  reserveTokens(remaining: number, requested: number): QuotaSnapshot {
    return { remaining: Math.max(0, remaining - requested) };
  }
}
