package example;

public class ApprovalGate {
    public boolean canPromote(boolean reviewed, boolean testsPassed) {
        return reviewed && testsPassed;
    }
}
