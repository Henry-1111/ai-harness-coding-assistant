package fixture

type RetryPolicy interface { NextDelay(attempt int) int }

func RetryAfterFailure(policy RetryPolicy, attempt int) int {
    return policy.NextDelay(attempt)
}
