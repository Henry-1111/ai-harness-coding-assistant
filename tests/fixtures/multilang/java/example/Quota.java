package example;

public class Quota {
    public int reserveTokens(int available, int requested) {
        return Math.max(available - requested, 0);
    }
}
