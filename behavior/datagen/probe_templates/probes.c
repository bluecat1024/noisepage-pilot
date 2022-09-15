#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

typedef struct pid_key {
    u64 id;
    u64 slot;
} pid_key_t;

BPF_HASH(trace, u32, u64, MAX_PID);
BPF_HASH(start, u32, u64, MAX_PID);
BPF_HISTOGRAM(dist);

void qss_ExecutorStart(struct pt_regs* ctx)
{
    u64 trace_ind = 1;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    trace.update(&tgid, &trace_ind);
}

void qss_ExecutorEnd(struct pt_regs* ctx)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32;
    trace.delete(&tgid);
}

static inline void store_start(u32 tgid, u32 pid, u64 ts)
{
    if (tgid != TARGET_PID)
        return;

    u64 *tsp = trace.lookup(&tgid);
    if (tsp == 0)
        return;

    start.update(&pid, &ts);
}

static inline void update_hist(u32 tgid, u32 pid, u64 ts)
{
    if (tgid != TARGET_PID)
        return;

    u64 *tsp = start.lookup(&pid);
    if (tsp == 0)
        return;

    if (ts < *tsp) {
        // Probably a clock issue where the recorded on-CPU event had a
        // timestamp later than the recorded off-CPU event, or vice versa.
        return;
    }

    u64 delta = (ts - *tsp) / 1000;
    dist.increment(bpf_log2l(delta));
}

int sched_switch(struct pt_regs *ctx, struct task_struct *prev)
{
    u64 ts = bpf_ktime_get_ns();
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 tgid = pid_tgid >> 32, pid = pid_tgid;

    u32 prev_pid = prev->pid;
    u32 prev_tgid = prev->tgid;
    store_start(prev_tgid, prev_pid, ts);

BAIL:
    update_hist(tgid, pid, ts);
    return 0;
}
