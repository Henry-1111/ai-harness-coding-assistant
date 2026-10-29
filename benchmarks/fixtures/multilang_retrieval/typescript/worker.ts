import { QuotaLedger } from './quota';
export function dispatchWork(ledger: QuotaLedger, amount: number): number {
  return ledger.reserveTokens(100, amount).remaining;
}
