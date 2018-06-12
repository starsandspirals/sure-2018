# Set size of output image
set terminal png size 1200,800 enhanced font 'Verdana,30'
#set key font ",30"
set xtics font "Verdana,20" 
set ytics font "Verdana,20" 

# Set name of output file.  The quotes are required
set output "PreyPredator.png"

# Set how the data is separated in the file
set datafile separator ","

set xlabel 'Iteration'                              # x-axis label
set ylabel 'Population'                             # y-axis label

set title "Predator-Prey model Agent Count"

# color definitions
set border linewidth 3

set tics nomirror

#set ytics format "{ %.0s}"
#set xtics format "{/:Bold %.0s}"

# logarithmic scale
#set logscale y

# key/legend
unset key
#set key top left
#set key horizontal center below maxrows 4
#set key box
set key bmargin 

set lmargin 10

# grid
set grid

plot "PreyPred_Count.csv" using ($0+1):2 with linespoints pt 6 lt 1 lw 3 lc 8 t "Prey", \
"" using ($0+1):4  w linespoints pt 3 lt 1 lw 3 lc 7  t  "Predator", "" using ($0+1):6  w linespoints pt 9 lt 1 lw 3 lc 2  t  "Grass" 
