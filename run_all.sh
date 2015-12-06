rm MIPS4WINNERS.out
for x in {1..621}; do 
    python algorithm.py instances/$x.in >> MIPS4WINNERS.out
done
