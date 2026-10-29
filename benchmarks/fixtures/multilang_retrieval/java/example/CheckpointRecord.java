package example;

public class CheckpointRecord {
    private final String taskId;
    private final long sequence;

    public CheckpointRecord(String taskId, long sequence) {
        this.taskId = taskId;
        this.sequence = sequence;
    }

    public boolean isCurrent(long latestSequence) {
        return sequence == latestSequence;
    }
}
