
function seq=logseq(N)
seq=1:9;
seq=[seq 10:2:49];
seq=[seq 50:5:99];
seq=[seq 100:20:499];
seq=[seq 500:50:999];
seq=[seq 1000:100:9999];
seq=[seq 10000:500:N];
seq=[seq(seq<N) N];
end
