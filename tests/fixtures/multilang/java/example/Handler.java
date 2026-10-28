package example;

import example.Quota;

public class Handler {
    private final Quota quota = new Quota();

    public Quota quota() {
        return quota;
    }
}
