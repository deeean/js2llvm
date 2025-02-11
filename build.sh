opt -O3 output.ll -o optimized.ll
llc ./optimized.ll -o output.s
gcc -O3 output.s
time ./a.out
