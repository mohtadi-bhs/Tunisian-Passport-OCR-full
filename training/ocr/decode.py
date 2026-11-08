def ctc_greedy(logits, char_map_rev):
    preds = logits.argmax(2)
    out = []
    for p in preds.permute(1, 0):
        seq = []
        prev = -1
        for t in p:
            t = t.item()
            if t == 0:
                prev = t
                continue
            if t == prev:
                prev = t
                continue
            seq.append(char_map_rev.get(t, ""))
            prev = t
        out.append("".join(seq))
    return out
