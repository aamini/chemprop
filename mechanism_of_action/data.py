from chembl_webresource_client.new_client import new_client as chembl_client
from typing import Dict, List, Set


class Target:
    """A molecule targeted by a Drug."""
    def __init__(self,
                 chembl_id: str,
                 organism: str,
                 go_terms: Dict[str, Set[str]] = None):
        """
        :param str chembl_id: a unique identifier for ChEMBL.
        :param str organism: the organism the Target is present in.
        :param Dict[str, Set[str]] go_terms: GO terms linked to the Target.
        """
        self.chembl_id = chembl_id
        self.organism = organism

        if go_terms:
            self.go_terms = go_terms
        else:
            self.go_terms = self.get_go_terms()

    def get_go_terms(self) -> List[str]:
        """Get the go terms associated with a particular target.

        :returns List[str]: a list of GO terms associated with the target.
        """
        targets = chembl_client.target.filter(
            target_chembl_id=self.chembl_id).only(['target_components'])

        if len(targets) == 0:
            raise ValueError(
                f'No target found for chEMBL ID: {self.chembl_id}')

        target = targets[0]
        go_terms = {'GoComponent': set(),
                    'GoFunction': set(),
                    'GoProcess': set()}

        for component in target['target_components']:
            xrefs = component['target_component_xrefs']
            for xref in xrefs:
                if xref['xref_src_db'] not in ['GoComponent',
                                               'GoFunction',
                                               'GoProcess']:
                    continue

                go_terms[xref['xref_src_db']].add(xref['xref_id'])

        return go_terms


class Drug:
    """An individual molecule targeting function's of an organism's cells."""
    def __init__(self,
                 chembl_id: str = None,
                 smiles: str = None):
        """
        :param str chembl_id: a unique identifier for ChEMBL.
        :param str smiles: the SMILES string of the Drug.
        :raises ValueError: if no molecule is found for given info.
        """
        if not smiles and not chembl_id:
            raise ValueError('No SMILES string or chEMBL id provided.')

        if not chembl_id:
            self.smiles = smiles

            molecules = chembl_client.molecule.filter(
                molecule_structures__canonical_smiles__flexmatch=smiles).only(
                    ['molecule_chembl_id'])

            if len(molecules) == 0:
                raise ValueError(f'No molecule found for SMILES: {smiles}')

            self.chembl_id = molecules[0]['molecule_chembl_id']
        elif not smiles:
            self.chembl_id = chembl_id

            molecules = chembl_client.molecule.get(chembl_id).only(
                ['molecular_structures'])

            if len(molecules) == 0:
                raise ValueError(
                    f'No molecule found for chEMBL ID: {chembl_id}')

            molecular_structures = molecules[0]['molecular_structures']
            self.smiles = molecular_structures['canonical_smiles']
        else:
            self.smiles = smiles
            self.chembl_id = chembl_id

    def get_targets(self) -> List[Target]:
        """
        :returns List[Target]: the molecules targeted by the Drug.
        """
        activities = chembl_client.activity.filter(
            molecule_chembl_id__in=[self.chembl_id]).only(['target_chembl_id',
                                                           'target_organism'])

        target_pairs = set()

        for activity in activities:
            target_pairs.add((activity['target_chembl_id'],
                              activity['target_organism']))

        return [Target(chembl_id, organism) for chembl_id,
                organism in target_pairs]
