import gemmi
# We will use the gemmi (https://pypi.org/project/gemmi/) software package
# for working with macromolecular models and the diffraction experiment.

# Assemble products of structure factors for each pair of atom type in a
# standard structure (rare types can be added later via exception handling)

class ScatteringLookup:
    """Calculate scattering by one of several methods depending on type of radiation.
    Argument gemmi_element is the gemmi object, from which the element name or atomic
    number will be fetched. All other default arguments will calculate non-anomalous
    scattering for an X-ray source. For documentation on these implementations, see
    https://gemmi.readthedocs.io/en/latest/scattering.html."""
    def __init__(self, scatterer="xray", anom=False, energy=None, wavelength=None, stol2=0.4):
        self.scatterer = scatterer
        self.anom = anom
        assert (not anom) or (energy is not None) or (wavelength is not None), \
            "either wavelength (Å) or energy (eV) needed for calculation of anomalous scattering factors"
        assert (energy is None) or (wavelength is None), \
            "conflict: please supply either wavelength (Å) or energy (eV), but not both"
        if wavelength is not None:
            energy = gemmi.hc/wavelength
        self.energy = energy
        self.stol2 = stol2 # stol2 is (sin(theta)/lambda)^2

        # Set up lookup table for scattering factors (can be updated in place by the
        # methods that use it if an element not listed here is encountered)
        self.single_element_SFs = {
            element: self.calculate_sf(element) for element in ('C','N','O','H','S')
        }
        self.pair_SFs = {
            (e1, e2):self.single_element_SFs[e1]*self.single_element_SFs[e2] \
            for e1 in self.single_element_SFs.keys() \
            for e2 in self.single_element_SFs.keys()
        }
        # There are duplicate entries for pairs in reversed order, e.g. both
        # ('C','O') and ('O','C'), but we will treat this as a feature, not a bug!

    def calculate_sf(self, element, stol2=None):
        """Calculate each atomic scattering factor (once). Element should be supplied
        as a string, as the 1-to-2-letter code. (sin(theta)/lambda)^2 can be
        overridden here but normally will be whatever was passed to __init__."""
        gemmi_element = gemmi.Element(element)
        match self.scatterer:
            case "xray":
                if self.anom:
                    # use Cromer-Liberman algorithm for calculating X-ray scattering
                    # factors with f' and f" contributions
                    get_coeffs = gemmi.cromer_liberman(z=gemmi_element.atomic_number, energy=self.energy)
                else:
                    # use X-ray scattering factors tabulated in the International Tables
                    # of Crystallography (1992) without anomalous scattering
                    get_coeffs = gemmi_element.it92
            case "electron":
                # use 5-Gaussian approximation of electron form factors as described in
                # the International Tables of Crystallography C (2011): table 4.3.2.2
                assert not self.anom, "anomalous scattering factors not available for electrons"
                get_coeffs = gemmi_element.c4322
            case "neutron":
                # use bound coherent scattering lengths from NCNR (1992) for neutrons
                assert not self.anom, "anomalous  scattering factors not available for neutrons"
                get_coeffs = gemmi_element.neutron92
        return get_coeffs.calculate_sf(stol2=(stol2 or self.stol2))

    def get_sf(self, element_pair):
        """Look up a product of two elemental scattering factors. Calcualte only on
        the first lookup. Pass a tuple element_pair of element code strings to this
        method, e.g. ('O','S')."""
        try:
            return self.pair_SFs[element_pair]
        except KeyError:
            # throw and catch an exception when the pair's scattering factor
            # product is not in the lookup table already, and try to add it.
            sfs = []
            for elem in element_pair:
                try:
                    # first attempt lookup (should still work for one atom in most
                    # pairs where the other atom's scattering must be calculated)
                    sfs.append(self.single_element_SFs[elem])
                except KeyError:
                    try:
                        # calculate the first time encountering each uncommon element
                        sfs.append(self.calculate_sf(elem))
                    except AttributeError:
                        # may still fail on especially heavy elements such as Nh
                        assert False, f"Failed to calculate scattering for element {elem}"
            # compute product of scattering factors and update lookup
            self.pair_SFs[element_pair] = scattering = sfs[0] * sfs[1]
            return scattering

class PattersonSim:
    """Determine the expected positions and intensities of peaks in the
    Patterson map for an arbitrary macromolecule based on its structure
    and crystal symmetry."""
    def __init__(self, model_path):
        # load atomistic model from PDB or mmCIF file
        self.model = gemmi.read_structure(model_path)
        self.get_atom_pairs()
        self.scattering_lookup = ScatteringLookup()
        # TODO: initialize lookup with args drawn from the model file header
        # and anomalous if possible
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
            scattering = self.scattering_lookup.get_sf((atom1.element.name, atom2.element.name))

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
