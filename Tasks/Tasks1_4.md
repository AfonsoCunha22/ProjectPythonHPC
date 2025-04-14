# Tasks 1 to 4
### Author - Afonso Cunha s233177
In this section, we intend to create and initialize the project by learning the data, it's structure and how is it handled. This way, we've combined the tasks 1 to 4 that were mainly focused on this initial part and are being answered on this document.

## Task 1 - Familiarize yourself with the data
> Load and visualize the input data for a few floorplans using a seperate Python script, Jupyter notebook or your preferred tool
## Task 2 - Familiarize yourself with the provided script
> Run and time the reference implementation for a
small subset of floorplans (e.g., 10 - 20). How long do you estimate it would take to process all
the floorplans? Perform the timing as a batch job so you get relieable results.

By the simulations we ran, we saw that, for 20 floorplans, we got a real time of 4 minutes and 15.868 seconds, user	time of 4 ninutes and 14.309 seconds and a system time of 0.207 seconds. 

Focusing on our wall-clock time (real) as it provides how long the job actually took, we know that:

$${
\text{real time} = 4\, \text{minutes} + \frac{15.868}{60}\, \text{minutes} = 4.2645\, \text{minutes}
}$$
  
Nonetheless, our focus is to estimate total runtime $( T_{\text{total}} )$, considering our $4571$ floorplans based on the runtime for a sample of \( N \) buildings:

$${
T_{\text{total}} = \frac{T_{\text{sample}}}{N} \times 4571
}$$


Plugging in the numbers:

$$
T_{\text{total}} = \frac{4.2645}{20} \times 4571 = 975.62\, \text{minutes}$$


$${
\frac{976}{60} \approx \textbf{16.27 hours}
}$$
## Task 3 - Visualize the simulation results for a few floorplans
## Task 4 - Profile the reference jacobi function using kernprof.
> Explain the different parts of the function and how much time each part takes