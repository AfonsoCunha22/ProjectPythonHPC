elapsed_minutes = 4 + 15.868 / 60 
total_floorplans = 4571
sample_floorplans = 20

estimated_total = (elapsed_minutes / sample_floorplans) * total_floorplans
print(f"Estimated runtime: {estimated_total:.2f} minutes")