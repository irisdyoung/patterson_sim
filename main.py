import gemmi
# We will use the gemmi (https://pypi.org/project/gemmi/) software package
# for working with macromolecular models and the diffraction experiment.

# Assemble products of structure factors for each pair of atom type in a
# standard structure (rare types can be added later via exception handling)

# Standardize here how to get scattering factor -- reused below
def get_sf(gemmi_element, scatterer="xray", anom=False, energy=None, wavelength=None, stol2=0.4):
    """Calculate scattering by one of several methods depending on type of radiation.
    Argument gemmi_element is the gemmi object, from which the element name or atomic
    number will be fetched. All other default arguments will calculate non-anomalous
    scattering for an X-ray source. For documentation on these implementations, see
    https://gemmi.readthedocs.io/en/latest/scattering.html."""
    def get_energy():
        assert (energy is not None) or (wavelength is not None), "either wavelength (Å) or energy (eV) needed for calculation of scattering factor"
        assert (energy is None) or (wavelength is None), "conflict: please supply either wavelength (Å) or energy (eV), but not both"
        if wavelength is not None:
            energy = gemmi.hc/wavelength
        return energy
    match scatterer:
        case "xray":
            if anom:
                # use Cromer-Liberman algorithm for calculating X-ray scattering
                # factors with f' and f" contributions
                coeffs = gemmi.cromer_liberman(z=gemmi_element.atomic_number, energy=get_energy())
            else:
                # use X-ray scattering factors tabulated in the International Tables
                # of Crystallography (1992) without anomalous scattering
                coeffs = gemmi.Element(gemmi_element).it92
        case "electron":
            # use 5-Gaussian approximation of electron form factors as described in
            # the International Tables of Crystallography C (2011): table 4.3.2.2
            assert not anom, "anomalous scattering factors not available for electrons"
            coeffs = gemmi_element.c4322
        case "neutron":
            # use bound coherent scattering lengths from NCNR (1992) for neutrons
            assert not anom, "anomalous  scattering factors not available for neutrons"
            coeffs = gemmi_element.neutron92
    return coeffs.calculate_sf(stol2=stol2)

# Set up lookup table for scattering factors (can be updated in place by the methods
# that use it if an element not listed here is encountered)
single_element_SFs = {
    element: get_sf(element) for element in ('C','N','O','H','S')
}
pair_SFs = {
    (e1, e2):single_element_SFs[e1]*single_element_SFs[e2] \
    for e1 in single_element_SFs.keys() \
    for e2 in single_element_SFs.keys()
}
# There are duplicate entries for pairs in reversed order, e.g. both
# ('C','O') and ('O','C'), but we will treat this as a feature, not a bug!

class PattersonSim:
    """Determine the expected positions and intensities of peaks in the
    Patterson map for an arbitrary macromolecule based on its structure
    and crystal symmetry."""
    def __init__(self, model_path):
        # load atomistic model from PDB or mmCIF file
        self.model = gemmi.read_structure(model_path)
        self.get_atom_pairs()
        self.get_peaks()

    def get_atom_pairs(self, include_h=True, include_water=True):
        """Take inventory of all possible pairings of atoms, each of which will
        contribute a peak to the (simulated) Patterson map."""

        print("Aggregating atoms...")
        # First get single, flat list of all atoms
        self.atoms = []
        for assembly in self.model:
            for chain in assembly:
                for residue in chain:
                    if not include_water and residue.name in ('HOH', 'WAT'): continue
                    for atom in residue:
                        if not include_h and atom.element.name == 'H': continue
                        self.atoms.append((assembly.num, chain.name, residue.name, atom))
        print("Generating atom pairs...")
        # Then generate all possible pairs (TODO: parallelize)
        self.atom_pairs = [
            (atom1, atom2) for atom1 in self.atoms for atom2 in self.atoms
        ]

    def get_peaks(self, include_h=True):
        """Calculate a Patterson peak corresponding to each pair of atoms."""
        # TODO: parallelize by evenly distributing atom pairs among procs
        self.patterson_peaks = []

        print("Begin calculating vectors and peak heights...")
        for (asb1, ch1, res1, atom1), (asb2, ch2, res2, atom2) in self.atom_pairs:
            # skip hydrogens if requested
            if not include_h and (atom1.element.name == 'H' or atom2.element.name == 'H'): continue

            # get the product of the scattering factors from the two atoms
            try:
                scattering = pair_SFs[(atom1.element.name, atom2.element.name)]
            except KeyError:
                # throw and catch an exception when the pair's scattering factor
                # product is not in the lookup table already, and add it if it can
                # be calculated from the ITC. Warn, but continue, if it can't be
                # calculated from the table values.
                try:
                    scattering = get_sf(atom1.element.name) * get_sf(atom2.element.name)
                except AttributeError:
                    # possible for metals and other heavier elements, e.g. Nh
                    print(f"Warning: ignoring atom pair {atom1.element.name}, {atom2.element.name}")
                    continue
                # update lookup and continue
                pair_SFs[(atom1.element.name, atom2.element.name)] = scattering

            # scale to account for occupancy
            scattering *= (atom1.occ * atom2.occ)

            # TODO: B-factor
            
            # get the vector between the atoms, in Ångstroms, relative to the
            # unit cell origin
            vec = atom2.pos - atom1.pos
            self.patterson_peaks.append((((asb1, ch1, res1, atom1), (asb2, ch2, res2, atom2)), vec, scattering))

            # TODO: put it in some format readable by Coot so we can look at it

if __name__ == "__main__":
    import sys, os
    path = sys.argv[1]
    assert os.path.exists(path), f"invalid PDB path {path}"
    sim = PattersonSim(path)
    print(f'Calculated {len(sim.patterson_peaks)} peaks for model {path}.')
    print('First ten peaks:')
    for i in range(10):
        ((asb1, ch1, res1, atom1), (asb2, ch2, res2, atom2)), vec, height = sim.patterson_peaks[i]
        print(f'{asb1}/{ch1}/{res1}/{atom1.name:4s} - {asb2}/{ch2}/{res2}/{atom2.name:4s}\tu,v,w = ({vec.x:6.3f}, {vec.y:6.3f}, {vec.z:6.3f}),\tpeak height: {height:.3f}')
