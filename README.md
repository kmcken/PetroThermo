# PetroThermo
Thermodynamics models for CSM PEGN511: Advanced Thermodynamics and Petroleum Fluids Phase Behavior

For Homework #2, the main file generates the figures for DeltaH and DeltaS.

If you want to evaluate the departure functions at specific values, you will need to edit main.py manually.

Reference the departure functions as the following:
Single.departure_H(temp, press, temp_crit, press_crit, acentric_factor)
Single.departure_S(temp, press, temp_crit, press_crit, acentric_factor)
Single.departure_U(temp, press, temp_crit, press_crit, acentric_factor)
Single.departure_A(temp, press, temp_crit, press_crit, acentric_factor)
Single.departure_G(temp, press, temp_crit, press_crit, acentric_factor)

To display the value, using the "print" command. For example:
print(Single.departure_H(temp, press, temp_crit, press_crit, acentric_factor))


Please contact me if you have any questions.
Kirt McKenna
kmckenna@mymail.mines.edu