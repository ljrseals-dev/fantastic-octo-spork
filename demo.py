from vocal_studio import *
lead = [Note(p, i*0.6, 0.55 if i%4 else 1.1, 0.8, l) for i,(p,l) in enumerate(
    [(60,"la"),(62,"la"),(64,"lo"),(67,"lee"),(65,"la"),(64,"lo"),(62,"la"),(60,"loo")])]
for i,n in enumerate(lead): n.start = sum(0.6 for _ in range(i)) + (0.5 if i>=4 else 0)
key = Key(0, "major")
s = Session(tempo=100, key_tonic=0, key_mode="major")
s.tracks = build_stack(lead, key, parts=4)
s.tracks += build_doubles(s.track("Lead"), 2) + build_backgrounds(lead, key, "airy")
m = VocalMixer(s)
print({k: str(v) for k, v in m.export("out").items()})
s.save("out/session.json")
