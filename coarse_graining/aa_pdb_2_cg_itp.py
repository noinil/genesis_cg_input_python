#!/usr/bin/env python

###############################################################################
#                                    README
# 
# This program read PDB structures and prepare toppology and coordinate files
# for CG MD simulations in Genesis.
#
# PDB format:
# 1. Atoms startswith "ATOM  "
# 2. Chains should end with "TER" and have different IDs
# 
###############################################################################

import numpy as np
import argparse
from tqdm import tqdm

###########################################################################
#                          Force Field Parameters                         #
###########################################################################
#      ____   _    ____      _    __  __ _____ _____ _____ ____  ____
#     |  _ \ / \  |  _ \    / \  |  \/  | ____|_   _| ____|  _ \/ ___|
#     | |_) / _ \ | |_) |  / _ \ | |\/| |  _|   | | |  _| | |_) \___ \
#     |  __/ ___ \|  _ <  / ___ \| |  | | |___  | | | |___|  _ < ___) |
#     |_| /_/   \_\_| \_\/_/   \_\_|  |_|_____| |_| |_____|_| \_\____/
#
###########################################################################


# ==================
# Physical Constants
# ==================
CAL2JOU = 4.184

# =====================================
# General Parameters: Mass, Charge, ...
# =====================================

ATOM_MASS_DICT = {
    'C' : 12.011,
    'N' : 14.001,
    'O' : 15.999,
    'P' : 30.974,
    'S' : 32.065,
    'H' :  1.008
}

RES_MASS_DICT = {
    "ALA" :  71.09,
    "ARG" : 156.19,
    "ASN" : 114.11,
    "ASP" : 115.09,
    "CYS" : 103.15,
    "CYM" : 103.15,
    "CYT" : 103.15,
    "GLN" : 128.14,
    "GLU" : 129.12,
    "GLY" :  57.05,
    "HIS" : 137.14,
    "ILE" : 113.16,
    "LEU" : 113.16,
    "LYS" : 128.17,
    "MET" : 131.19,
    "PHE" : 147.18,
    "PRO" :  97.12,
    "SER" :  87.08,
    "THR" : 101.11,
    "TRP" : 186.21,
    "TYR" : 163.18,
    "VAL" :  99.14,
    "DA"  : 134.10,
    "DC"  : 110.10,
    "DG"  : 150.10,
    "DT"  : 125.10,
    "DP"  :  94.97,
    "DS"  :  83.11,
    "RA"  : 134.10,
    "RC"  : 110.10,
    "RG"  : 150.10,
    "RU"  : 111.10,
    "RP"  :  62.97,
    "RS"  : 131.11
}

RES_CHARGE_DICT = {
    "ALA" :  0.0,
    "ARG" :  1.0,
    "ASN" :  0.0,
    "ASP" : -1.0,
    "CYS" :  0.0,
    "CYM" :  0.0,
    "CYT" :  0.0,
    "GLN" :  0.0,
    "GLU" : -1.0,
    "GLY" :  0.0,
    "HIS" :  0.0,
    "ILE" :  0.0,
    "LEU" :  0.0,
    "LYS" :  1.0,
    "MET" :  0.0,
    "PHE" :  0.0,
    "PRO" :  0.0,
    "SER" :  0.0,
    "THR" :  0.0,
    "TRP" :  0.0,
    "TYR" :  0.0,
    "VAL" :  0.0,
    "DA"  :  0.0,
    "DC"  :  0.0,
    "DG"  :  0.0,
    "DT"  :  0.0,
    "DP"  : -0.6,
    "DS"  :  0.0,
    "RA"  :  0.0,
    "RC"  :  0.0,
    "RG"  :  0.0,
    "RU"  :  0.0,
    "RP"  : -1.0,
    "RS"  :  0.0
}

RES_SHORTNAME_DICT = {
    "ALA" : "A",
    "ARG" : "R",
    "ASN" : "N",
    "ASP" : "D",
    "CYS" : "C",
    "CYM" : "C",
    "CYT" : "C",
    "GLN" : "Q",
    "GLU" : "E",
    "GLY" : "G",
    "HIS" : "H",
    "ILE" : "I",
    "LEU" : "L",
    "LYS" : "K",
    "MET" : "M",
    "PHE" : "F",
    "PRO" : "P",
    "SER" : "S",
    "THR" : "T",
    "TRP" : "W",
    "TYR" : "Y",
    "VAL" : "V",
    "DA"  : "A",
    "DC"  : "C",
    "DG"  : "G",
    "DT"  : "T",
    "RA"  : "A",
    "RC"  : "C",
    "RG"  : "G",
    "RU"  : "U"
}

RES_NAME_SET_PROTEIN = (
    "ALA", "ARG", "ASN", "ASP",
    "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS",
    "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    "CYM", "CYT")

RES_NAME_SET_DNA = ("DA", "DC", "DG", "DT")

RES_NAME_SET_RNA = ("RA", "RC", "RG", "RU")

# DNA CG residue atom names
ATOM_NAME_SET_DP = ("P", "OP1", "OP2", "O5'", "O1P", "O2P")
ATOM_NAME_SET_DS = ("C5'", "C4'", "C3'", "C2'", "C1'", "O4'", "O2'")

# RNA CG residue atom names
ATOM_NAME_SET_RP = ("P", "OP1", "OP2", "O1P", "O2P")
ATOM_NAME_SET_RS = ("C5'", "C4'", "C3'", "C2'", "C1'", "O5'", "O4'", "O3'", "O2'")

# ==============
# Molecule Types
# ==============

MOL_DNA     = 1
MOL_RNA     = 2
MOL_PROTEIN = 3
MOL_OTHER   = 4
MOL_TYPE_LIST = ["DNA", "RNA", "protein", "other", "unknown"]

# ===============================
# Protein AICG2+ Model Parameters
# ===============================

# AICG2+ bond force constant
AICG_BOND_K               = 110.40 * CAL2JOU * 100.0 * 2.0
# AICG2+ sigma for Gaussian angle
AICG_13_SIGMA             = 0.15 * 0.1  # nm
# AICG2+ sigma for Gaussian dihedral
AICG_14_SIGMA             = 0.15        # Rad ??
# AICG2+ atomistic contact cutoff
AICG_GO_ATOMIC_CUTOFF     = 6.5
# AICG2+ pairwise interaction cutoff
AICG_ATOMIC_CUTOFF        = 5.0
# AICG2+ hydrogen bond cutoff
AICG_HYDROGEN_BOND_CUTOFF = 3.2
# AICG2+ salt bridge cutoff
AICG_SALT_BRIDGE_CUTOFF   = 3.5
# AICG2+ energy cutoffs
AICG_ENE_UPPER_LIM        = -0.5
AICG_ENE_LOWER_LIM        = -5.0
# average and general AICG2+ energy values
AICG_13_AVE               = 1.72
AICG_14_AVE               = 1.23
AICG_CONTACT_AVE          = 0.55
AICG_13_GEN               = 1.11
AICG_14_GEN               = 0.87
AICG_CONTACT_GEN          = 0.32

# AICG2+ pairwise interaction pairs
AICG_ITYPE_BB_HB = 1  # B-B hydrogen bonds
AICG_ITYPE_BB_DA = 2  # B-B donor-accetor contacts
AICG_ITYPE_BB_CX = 3  # B-B carbon-X contacts
AICG_ITYPE_BB_XX = 4  # B-B other
AICG_ITYPE_SS_HB = 5  # S-S hydrogen bonds
AICG_ITYPE_SS_SB = 6  # S-S salty bridge
AICG_ITYPE_SS_DA = 7  # S-S donor-accetor contacts
AICG_ITYPE_SS_CX = 8  # S-S carbon-X contacts
AICG_ITYPE_SS_QX = 9  # S-S charge-X contacts
AICG_ITYPE_SS_XX = 10 # S-S other
AICG_ITYPE_SB_HB = 11 # S-B hydrogen bonds
AICG_ITYPE_SB_DA = 12 # S-B donor-accetor contacts
AICG_ITYPE_SB_CX = 13 # S-B carbon-X contacts
AICG_ITYPE_SB_QX = 14 # S-B charge-X contacts
AICG_ITYPE_SB_XX = 15 # S-B other
AICG_ITYPE_LR_CT = 16 # long range contacts
AICG_ITYPE_OFFST = 0  # offset

AICG_PAIRWISE_ENERGY = np.zeros(17)
AICG_PAIRWISE_ENERGY[AICG_ITYPE_BB_HB] = - 1.4247 # B-B hydrogen bonds
AICG_PAIRWISE_ENERGY[AICG_ITYPE_BB_DA] = - 0.4921 # B-B donor-accetor contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_BB_CX] = - 0.2404 # B-B carbon-X contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_BB_XX] = - 0.1035 # B-B other
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_HB] = - 5.7267 # S-S hydrogen bonds
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_SB] = -12.4878 # S-S salty bridge
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_DA] = - 0.0308 # S-S donor-accetor contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_CX] = - 0.1113 # S-S carbon-X contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_QX] = - 0.2168 # S-S charge-X contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SS_XX] =   0.2306 # S-S other
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SB_HB] = - 3.4819 # S-B hydrogen bonds
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SB_DA] = - 0.1809 # S-B donor-accetor contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SB_CX] = - 0.1209 # S-B carbon-X contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SB_QX] = - 0.2984 # S-B charge-X contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_SB_XX] = - 0.0487 # S-B other
AICG_PAIRWISE_ENERGY[AICG_ITYPE_LR_CT] = - 0.0395 # long range contacts
AICG_PAIRWISE_ENERGY[AICG_ITYPE_OFFST] = - 0.1051 # offset

# ============================
# DNA 3SPN.2C Model Parameters
# ============================

# 3SPN.2C bond force constant
DNA3SPN_BOND_K_2    = 60.0 * 2
# 3SPN.2C force constant for Gaussian dihedral
DNA3SPN_DIH_G_K     = 7.0
# 3SPN.2C sigma for Gaussian dihedral
DNA3SPN_DIH_G_SIGMA = 0.3
# 3SPN.2C force constant for Gaussian dihedral
DNA3SPN_DIH_P_K     = 2.0

# ====================================
# RNA Structure-based Model Parameters
# ====================================

# RNA atomistic contact cutoff
RNA_GO_ATOMIC_CUTOFF  = 5.5
# RNA stacking interaction dihedral cutoff
RNA_STACK_DIH_CUTOFF  = 40.0
# RNA stacking interaction distance cutoff
RNA_STACK_DIST_CUTOFF = 6.0
# RNA stacking interaction epsilon
RNA_STACK_EPSILON     = 2.06
# RNA base pairing epsilon
RNA_BPAIR_EPSILON_2HB = 2.94
RNA_BPAIR_EPSILON_3HB = 5.37

RNA_BOND_K_LIST = {
    "PS" : 26.5,
    "SR" : 40.3,
    "SY" : 62.9,
    "SP" : 84.1
}
RNA_ANGLE_K_LIST = {
    "PSR" : 18.0,
    "PSY" : 22.8,
    "PSP" : 22.1,
    "SPS" : 47.8
}
RNA_DIHEDRAL_K_LIST = {
    "PSPS" : 1.64,
    "SPSR" : 1.88,
    "SPSY" : 2.82,
    "SPSP" : 2.98
}
RNA_PAIR_EPSILON_OTHER = {
    "SS" : 1.48,
    "BS" : 0.98,
    "SB" : 0.98,
    "BB" : 0.93
}

# =================
# PWMcos parameters
# =================
# PWMcos atomistic contact cutoff
PWMCOS_ATOMIC_CUTOFF    = 4.0

# ======================
# Protein-RNA parameters
# ======================
# protein-RNA Go-term coefficient
PRO_RNA_GO_EPSILON_B    = 0.62
PRO_RNA_GO_EPSILON_S    = 0.74


# ====================
# GRO TOP File Options
# ====================

# "NREXCL" in "[moleculetype]"
MOL_NR_EXCL             = 3
# "CGNR" in "[atoms]"
AICG_ATOM_FUNC_NR       = 1
DNA3SPN_ATOM_FUNC_NR    = 1
RNA_ATOM_FUNC_NR        = 1
# "f" in "[bonds]"
AICG_BOND_FUNC_TYPE     = 1
DNA3SPN_BOND_FUNC2_TYPE = 1
DNA3SPN_BOND_FUNC4_TYPE = 21
RNA_BOND_FUNC_TYPE      = 1
# "f" in AICG-type "[angles]"
AICG_ANG_G_FUNC_TYPE    = 21
# "f" in Flexible-type "[angles]"
AICG_ANG_F_FUNC_TYPE    = 22
# "f" in DNA "[angles]"
DNA3SPN_ANG_FUNC_TYPE   = 1
# "f" in RNA "[angles]"
RNA_ANG_FUNC_TYPE       = 1
# "f" in AICG-type "[dihedral]"
AICG_DIH_G_FUNC_TYPE    = 21
# "f" in Flexible-type "[dihedral]"
AICG_DIH_F_FUNC_TYPE    = 22
# "f" in DNA Gaussian "[dihedral]"
DNA3SPN_DIH_G_FUNC_TYPE = 21
# "f" in DNA Periodic "[dihedral]"
DNA3SPN_DIH_P_FUNC_TYPE = 1
DNA3SPN_DIH_P_FUNC_PERI = 1
# "f" in RNA Periodic "[dihedral]"
RNA_DIH_FUNC_TYPE       = 1
# "f" in Go-contacts "[pairs]"
AICG_CONTACT_FUNC_TYPE  = 2
# "f" in RNA Go-contacts "[pairs]"
RNA_CONTACT_FUNC_TYPE   = 2
# "f" in pro-RNA Go-contacts "[pairs]"
RNP_CONTACT_FUNC_TYPE   = 2
# "f" in protein-DNA PWMcos "[pwmcos]"
PWMCOS_FUNC_TYPE        = 1



###############################################################################
#                                  Functions                                  #
###############################################################################
#          ____    _    ____ ___ ____   _____ _   _ _   _  ____
#         | __ )  / \  / ___|_ _/ ___| |  ___| | | | \ | |/ ___|
#         |  _ \ / _ \ \___ \| | |     | |_  | | | |  \| | |
#         | |_) / ___ \ ___) | | |___  |  _| | |_| | |\  | |___
#         |____/_/   \_\____/___\____| |_|    \___/|_| \_|\____|
#
###############################################################################

# ===================
# Geometric Functions
# ===================

# --------
# Distance
# --------

def compute_distance(coor1, coor2):
    d = coor1 - coor2
    return np.linalg.norm(d)

# -----
# Angle
# -----

def compute_angle(coor1, coor2, coor3):
    v1 = coor1 - coor2
    v2 = coor3 - coor2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return np.arccos( np.dot(v1, v2) / n1 / n2) / np.pi * 180.0

def compute_vec_angle(vec1, vec2):
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    return np.arccos( np.dot(vec1, vec2) / n1 / n2) / np.pi * 180.0


# --------
# Dihedral
# --------

def compute_dihedral(coor1, coor2, coor3, coor4):
    v12   = coor2 - coor1
    v23   = coor3 - coor2
    v34   = coor4 - coor3
    c123  = np.cross(v12, v23)
    c234  = np.cross(v23, v34)
    nc123 = np.linalg.norm(c123)
    nc234 = np.linalg.norm(c234)
    dih   = np.arccos( np.dot(c123, c234) / nc123 / nc234)
    c1234 = np.cross(c123, c234)
    judge = np.dot(c1234, v23)
    dih   = dih if judge > 0 else -dih
    return dih / np.pi * 180.0


# --------------
# Center of mass
# --------------

def compute_center_of_mass(atom_indices, atom_names, atom_coors):
    total_mass      = 0
    tmp_coor        = np.zeros(3)
    for i in atom_indices:
        a_mass      = ATOM_MASS_DICT[atom_names[i][1]]
        a_coor      = atom_coors[i, :]
        total_mass += a_mass
        tmp_coor   += a_coor * a_mass
    com = tmp_coor / total_mass
    return com


# ===============================
# Structural Biological Functions
# ===============================

# --------------------
# AICG2+ Protein Model
# --------------------

def is_protein_backbone(atom_name):
    if atom_name in ("N", "C", "O", "OXT", "CA"):
        return True
    return False

def is_protein_hb_donor(atom_name, res_name):
    if atom_name[0] == 'N':
        return True
    elif atom_name[0] == 'S' and res_name == "CYS":
        return True
    elif atom_name[0] == 'O':
        if  ( res_name == "SER" and atom_name == "OG"  ) or \
            ( res_name == "THR" and atom_name == "OG1" ) or \
            ( res_name == "TYR" and atom_name == "OH"  ):
            return True
    return False

def is_protein_hb_acceptor(atom_name):
    if atom_name[0] == 'O' or atom_name[0] == 'S':
        return True
    return False

def is_protein_cation(atom_name, res_name):
    if atom_name[0] == 'N':
        if  ( res_name == "ARG" and atom_name == "NH1" ) or \
            ( res_name == "ARG" and atom_name == "NH2" ) or \
            ( res_name == "LYS" and atom_name == "NZ"  ):
            return True
    return False

def is_protein_anion(atom_name, res_name):
    if atom_name[0] == 'O':
        if  ( res_name == "GLU" and atom_name == "OE1" ) or \
            ( res_name == "GLU" and atom_name == "OE2" ) or \
            ( res_name == "ASP" and atom_name == "OD1" ) or \
            ( res_name == "ASP" and atom_name == "OD2" ):
            return True
    return False

def is_protein_hb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
    if  is_protein_hb_acceptor(atom_name_1) and \
        is_protein_hb_donor(atom_name_2, res_name_2):
        return True
    elif is_protein_hb_acceptor(atom_name_2) and \
         is_protein_hb_donor(atom_name_1, res_name_1):
        return True
    return False

def is_protein_sb_pair(atom_name_1, res_name_1, atom_name_2, res_name_2):
    if  is_protein_cation(atom_name_1, res_name_1) and \
        is_protein_anion(atom_name_2,  res_name_2):
        return True
    elif is_protein_cation(atom_name_2, res_name_2) and \
         is_protein_anion(atom_name_1,  res_name_1):
        return True
    return False

def is_protein_nonsb_charge_pair(atom_name_1, res_name_1, atom_name_2, res_name_2)
    if  is_protein_cation(atom_name_1, res_name_1) or \
        is_protein_anion(atom_name_1,  res_name_1) or \
        is_protein_cation(atom_name_2, res_name_2) or \
        is_protein_anion(atom_name_2,  res_name_2):
        return True
    return False

def is_protein_go_contact(resid1, resid2, atom_names, atom_coors):
    for i in resid1.atoms:
        atom_name_1 = atom_names[i]
        if atom_name_1[0] == 'H':
            continue
        coor_1 = atom_coors[i, :]
        for j in resid2.atoms:
            atom_name_2 = atom_names[j]
            if atom_name_2[0] == 'H':
                continue
            coor_2  = atom_coors[j, :]
            dist_12 = compute_distance(coor_1, coor_2)
            if dist_12 < AICG_GO_ATOMIC_CUTOFF:
                return True
    return False

def count_aicg_atomic_contact(resid1, resid2, res_name_1, res_name_2, atom_names, atom_coors):
    contact_count                   = np.zeros(( 17, ), dtype=int)
    contact_count[AICG_ITYPE_OFFST] = 1
    num_short_range_contact         = 0
    for i in resid1.atoms:
        atom_name_1 = atom_names[i]
        if atom_name_1[0] == 'H':
            continue
        coor_1 = atom_coors[i, :]
        for j in resid2.atoms:
            atom_name_2     = atom_names[j]
            if atom_name_2[0] == 'H'
                continue
            coor_2          = atom_coors[j, :]
            dist_12         = compute_distance(coor_1, coor_2)

            is_hb           = is_protein_hb_pair           (atom_name_1, res_name_1, atom_name_2, res_name_2)
            is_sb           = is_protein_sb_pair           (atom_name_1, res_name_1, atom_name_2, res_name_2)
            is_nonsb_charge = is_protein_nonsb_charge_pair (atom_name_1, res_name_1, atom_name_2, res_name_2)
            is_1_backbone   = is_protein_backbone          (atom_name_1)
            is_2_backbone   = is_protein_backbone          (atom_name_2)
            if dist_12 < AICG_GO_ATOMIC_CUTOFF:
                contact_count[AICG_ITYPE_LR_CT] += 1
            if dist_12 < AICG_ATOMIC_CUTOFF:
                num_short_range_contact += 1
                if is_1_backbone and is_2_backbone:
                    if is_hb:
                        if dist_12 < AICG_HYDROGEN_BOND_CUTOFF:
                            contact_count[AICG_ITYPE_BB_HB] += 1
                        else:
                            contact_count[AICG_ITYPE_BB_DA] += 1
                    elif atom_name_1[0] == 'C' or atom_name_2[0] == 'C':
                        contact_count[AICG_ITYPE_BB_CX] += 1
                    else:
                        contact_count[AICG_ITYPE_BB_XX] += 1
                elif ( not is_1_backbone ) and ( not is_2_backbone ):
                    if is_hb:
                        if is_sb:
                            if dist_12 < AICG_SALT_BRIDGE_CUTOFF:
                                contact_count[AICG_ITYPE_SS_SB] += 1
                            else:
                                contact_count[AICG_ITYPE_SS_QX] += 1
                        elif dist_12 < AICG_HYDROGEN_BOND_CUTOFF:
                            contact_count[AICG_ITYPE_SS_HB] += 1
                        elif is_nonsb_charge:
                            contact_count[AICG_ITYPE_SS_QX] += 1
                        else:
                            contact_count[AICG_ITYPE_SS_DA] += 1
                    elif is_nonsb_charge:
                        contact_count[AICG_ITYPE_SS_QX] += 1
                    elif atom_name_1[0] == 'C' or atom_name_2[0] == 'C':
                        contact_count[AICG_ITYPE_SS_CX] += 1
                    else:
                        contact_count[AICG_ITYPE_SS_XX] += 1
                elif ( is_1_backbone and ( not is_2_backbone ) ) or \
                     ( is_2_backbone and ( not is_1_backbone ) ):
                    if is_hb:
                        if dist_12 < AICG_HYDROGEN_BOND_CUTOFF:
                            contact_count[AICG_ITYPE_SB_HB] += 1
                        elif is_nonsb_charge:
                            contact_count[AICG_ITYPE_SB_QX] += 1
                        else:
                            contact_count[AICG_ITYPE_SB_DA] += 1
                        end
                    elif is_nonsb_charge:
                        contact_count[AICG_ITYPE_SB_QX] += 1
                    elif atom_name_1[0] == 'C' or atom_name_2[0] == 'C':
                        contact_count[AICG_ITYPE_SB_CX] += 1
                    else:
                        contact_count[AICG_ITYPE_SB_XX] += 1

    # control the number of long-range contacts
    if AICG_GO_ATOMIC_CUTOFF > AICG_ATOMIC_CUTOFF:
        contact_count[AICG_ITYPE_LR_CT] -= num_short_range_contact
    else:
        contact_count[AICG_ITYPE_LR_CT]  = 0

    # control the number of salty bridge
    if contact_count[AICG_ITYPE_SS_SB]  >= 2:
        contact_count[AICG_ITYPE_SS_QX] += contact_count[AICG_ITYPE_SS_SB] - 1
        contact_count[AICG_ITYPE_SS_SB]  = 1

    return contact_count

# -----------------
# 3SPN.2C DNA model
# -----------------

def get_DNA3SPN_angle_param(angle_type, base_step):
    # Base-Sugar-Phosphate
    BSP_params = {
        "AA" : 460, "AT" : 370, "AC" : 442, "AG" : 358,
        "TA" : 120, "TT" : 460, "TC" : 383, "TG" : 206,
        "CA" : 206, "CT" : 358, "CC" : 278, "CG" : 278,
        "GA" : 383, "GT" : 442, "GC" : 336, "GG" : 278
    }
    # Phosphate-Sugar-Base
    PSB_params = {
        "AA" : 460, "TA" : 120, "CA" : 206, "GA" : 383,
        "AT" : 370, "TT" : 460, "CT" : 358, "GT" : 442,
        "AC" : 442, "TC" : 383, "CC" : 278, "GC" : 336,
        "AG" : 358, "TG" : 206, "CG" : 278, "GG" : 278
    }
    # Phosphate-Sugar-Phosphate
    PSP_params = {
        "all" : 300
    }
    # Sugar-Phosphate-Sugar
    SPS_params = {
        "AA" : 355, "AT" : 147, "AC" : 464, "AG" : 368,
        "TA" : 230, "TT" : 355, "TC" : 442, "TG" : 273,
        "CA" : 273, "CT" : 368, "CC" : 165, "CG" : 478,
        "GA" : 442, "GT" : 464, "GC" : 228, "GG" : 165
    }
    angle_params = {
        "BSP" : BSP_params,
        "PSB" : PSB_params,
        "PSP" : PSP_params,
        "SPS" : SPS_params
    }

    return angle_params[angle_type][base_step]

# -------------------------
# RNA structure-based model
# -------------------------
def is_RNA_hydrogen_bond(atom_name_1, atom_name_2):
    special_atom_list = ['F', 'O', 'N']
    if atom_name_1 in special_atom_list and atom_name_2 in special_atom_list:
        return True
    return False

def compute_RNA_Go_contact(resid1, resid2, atom_names, atom_coors):
    hb_count = 0
    min_dist = 1e50
    for i in resid1.atoms:
        atom_name_1 = atom_names[i]
        if atom_name_1[0] == 'H':
            continue
        coor_1 = atom_coors[i, :]
        for j in resid2.atoms:
            atom_name_2 = atom_names[j]
            if atom_name_2[0] == 'H':
                continue
            coor_2 = atom_coors[j, :]
            dist_12 = compute_distance(coor_1, coor_2)
            if dist_12 < RNA_GO_ATOMIC_CUTOFF and is_RNA_hydrogen_bond(atom_name_1[0], atom_name_2[0]):
                hb_count += 1
            if dist_12 < min_dist
                min_dist = dist_12
    return (min_dist, hb_count)

# ------------------------
# protein-DNA interactions
# ------------------------

def is_PWMcos_contact(resid1, resid2, atom_names, atom_coors):
    for i in resid1.atoms:
        atom_name_1 = atom_names[i]
        if atom_name_1[0] == 'H':
            continue
        coor_1 = atom_coors[i, :]
        for j in resid2.atoms:
            atom_name_2 = atom_names[j]
            if atom_name_2[0] == 'H':
                continue
            coor_2  = atom_coors[j, :]
            dist_12 = compute_distance(coor_1, coor_2)
            if dist_12 < PWMCOS_ATOMIC_CUTOFF:
                return True
    return False

# ------------------------
# protein-RNA interactions
# ------------------------

def is_protein_RNA_go_contact(resid1, resid2, atom_names, atom_coors):
    for i in resid1.atoms:
        atom_name_1 = atom_names[i]
        if atom_name_1[0] == 'H':
            continue
        coor_1 = atom_coors[i, :]
        for j in resid2.atoms
            atom_name_2 = atom_names[j]
            if atom_name_2[0] == 'H':
                continue
            coor_2  = atom_coors[j, :]
            dist_12 = compute_distance(coor_1, coor_2)
            if dist_12 < AICG_GO_ATOMIC_CUTOFF:
                return True
    return False


# ------------------
# Other file formats
# ------------------

def read_modified_pfm(pfm_filename):
    pfm = {}
    with open(pfm_filename, 'r') as fin:
        for line in fin:
            words = split(line)
            if len(words) < 1:
                continue
            w1 = words[0]
            if w1 in "ACGT":
                local_list = []
                for p in words[1:]:
                    append(local_list, float(p))
                pfm[w1] = local_list
            elif w1 in ["CHAIN_A", "CHAIN_B"]:
                local_list = []
                for dna_id in words[1:]:
                    append(local_list, int(dna_id))
                pfm[w1] = local_list

    pfmat = np.array([pfm["A"],  pfm["C"], pfm["G"],  pfm["T"]])
    ppmat = pfmat / pfmat.sum(axis=0)
    pwmat0 = -np.log(ppmat)
    pwmat = pwmat0 - pwmat0.sum(axis=0) / 4

    return (pwmat, pfm["CHAIN_A"], pfm["CHAIN_B"])


# =============================
# Coarse-Graining Structures!!!
# =============================

class AAResidue:
    def __init__(self, name, atoms):
        self.name = name
        self.atoms = atoms

class AAChain:
    def __init__(self, chain_id, residues):
        self.chain_id = chain_id
        self.residues = residues


class CGResidue:
    def __init__(self, residue_index, residue_name, atom_name, atoms):
        self.res_idx = residue_index
        self.res_name = residue_name
        self.atm_name = atom_name
        self.atoms = atoms

class CGChain:
    def __init__(self, first, last, moltype):
        self.first = first
        self.last = last
        self.moltype = moltype


###############################################################################
#                            ____ ___  ____  _____
#                           / ___/ _ \|  _ \| ____|
#                          | |  | | | | |_) |  _|
#                          | |__| |_| |  _ <| |___
#                           \____\___/|_| \_\_____|
#
###############################################################################
# core function
def pdb_2_top(args):

    # -----------------
    # Parsing arguments
    # -----------------
    pdb_name                = args.pdb
    protein_charge_filename = args.respac
    scale_scheme            = args.aicg_scale
    gen_3spn_itp            = args.3spn_param
    gen_pwmcos_itp          = args.pwmcos
    pwmcos_gamma            = args.pwmcos_scale
    pwmcos_epsil            = args.pwmcos_shift
    pfm_filename            = args.pfm
    appendto_filename       = args.patch
    do_output_psf           = args.psf
    do_output_cgpdb         = args.cgpdb
    do_debug                = args.debug
    do_output_sequence      = args.show_sequence

    # ===============
    # Step 0: numbers
    # ===============

    aa_num_atom    = 0
    aa_num_residue = 0
    aa_num_chain   = 0

    num_chain_pro  = 0
    num_chain_DNA  = 0
    num_chain_RNA  = 0

    i_step         = 0

    # ================
    # Step 1: open PDB
    # ================
    i_step += 1
    print("============================================================")
    print("> Step {0>2d}: open PDB file.".format(i_step))

    aa_pdb_lines = []

    with open(pdb_name, "r") as fin_pdb:
        for line in fin_pdb:
            if line.startswith("ATOM"):
                aa_pdb_lines.append(line.ljust(80))
                aa_num_atom += 1
            elif line.startswith("TER") or line.startswith("END"):
                aa_pdb_lines.append(line.ljust(80))

    aa_atom_name  = ["    " for _ in range(aa_num_atom)]
    aa_coor       = np.zeros((aa_num_atom, 3))

    aa_residues   = []
    aa_chains     = []

    i_atom        = 0
    i_resid       = 0
    curr_resid    = None
    curr_chain    = None
    curr_rname    = "    "
    residue_name  = "    "
    chain_id      = '?'
    tmp_res_atoms = []
    tmp_chain_res = []
    
    for line in aa_pdb_lines:
        if line.startswith("TER") or line.startswith("END"):
            if len(tmp_res_atoms) > 0:
                aa_residues.append(AAResidue(residue_name, tmp_res_atoms[:]))
                tmp_res_atoms = []
            if len(tmp_chain_res) > 0
                aa_chains.append(AAChain(chain_id, tmp_chain_res[:]))
                tmp_chain_res = []
            continue

        i_atom        += 1
        atom_name      = line[12:16].strip()
        residue_name   = line[17:21].strip()
        chain_id       = line[21]
        atom_serial    = int   (line[6 :11])
        residue_serial = int   (line[22:26])
        coor_x         = float (line[30:38])
        coor_y         = float (line[38:46])
        coor_z         = float (line[46:54])

        aa_atom_name [i_atom - 1 ] = atom_name
        aa_coor      [i_atom - 1 ] = [ coor_x, coor_y, coor_z ]

        if residue_serial != curr_resid
            i_resid += 1
            tmp_chain_res.append(i_resid)
            curr_resid = residue_serial
            if len(tmp_res_atoms) > 0
                aa_residues.append(AAResidue(curr_rname, tmp_res_atoms[:]))
                tmp_res_atoms = []
            curr_rname = residue_name
        tmp_res_atoms.append(i_atom)

    aa_num_residue = len(aa_residues)
    aa_num_chain   = len(aa_chains)

    print("          > Number of atoms:    {0:>10d}".format(aa_num_atom))
    print("          > Number of residues: {0:>10d}".format(aa_num_residue))
    print("          > Number of chains:   {0:>10d}".format(aa_num_chain))

    # ===============================
    # Step 2: find out molecule types
    # ===============================
    i_step += 1
    print("============================================================")
    print("> Step {0:>2d}: set molecular types for every chain.".format(i_step))

    cg_num_particles = 0

    cg_chain_mol_types = np.zeros(aa_num_chain, dtype=int)
    cg_chain_length    = np.zeros(aa_num_chain, dtype=int)

    for i_chain range( aa_num_chain ):
        chain = aa_chains[i_chain]
        mol_type = -1
        for i_res in chain.residues:
            res_name = aa_residues[i_res].name
            if res_name in RES_NAME_SET_PROTEIN:
                tmp_mol_type = MOL_PROTEIN
            elif res_name in RES_NAME_SET_DNA:
                tmp_mol_type = MOL_DNA
            elif res_name in RES_NAME_SET_RNA:
                tmp_mol_type = MOL_RNA
            else:
                tmp_mol_type = MOL_OTHER
            if mol_type == -1:
                mol_type = tmp_mol_type
            elif tmp_mol_type != mol_type:
                errmsg = "BUG: Inconsistent residue types in chain {} ID - {} residue - {} : {} "
                print(errmsg.format(i_chain, chain.chain_id, i_res, res_name))
                exit()
        cg_chain_mol_types[i_chain] = mol_type
        n_res = len(chain.residues)
        if mol_type == MOL_DNA:
            n_particles = 3 * n_res - 1
            num_chain_DNA += 1
        elif mol_type == MOL_RNA:
            n_particles = 3 * n_res - 1
            num_chain_RNA += 1
        elif mol_type == MOL_PROTEIN:
            n_particles = n_res
            num_chain_pro += 1
        else:
            n_particles = 0
        cg_chain_length[i_chain] = n_particles
        cg_num_particles += n_particles
        print("          > Chain {0:>3d} | {1:>7} \n".format( i_chain, MOL_TYPE_LIST[ mol_type ] ))

    print("------------------------------------------------------------")
    print("          In total: {0:>5d} protein chains,\n".format(num_chain_pro))
    print("                    {0:>5d} DNA    strands,\n".format(num_chain_DNA))
    print("                    {0:>5d} RNA    strands.\n".format(num_chain_RNA))

    # ===========================
    # Step 3: Assign CG particles
    # ===========================
    i_step += 1
    print("============================================================")
    print("> Step {0:>2d}: assign coarse-grained particles.".format(i_step))

    cg_residues = []
    cg_chains   = []

    i_offset_cg_particle = 0
    i_offset_cg_residue  = 0

    for i_chain in range(aa_num_chain):
        chain    = aa_chains          [i_chain]
        mol_type = cg_chain_mol_types [i_chain]

        i_bead   = i_offset_cg_particle
        i_resi   = i_offset_cg_residue

        if mol_type == MOL_PROTEIN:
            for i_res in chain.residues:
                cg_idx = []
                res_name = aa_residues[i_res].name
                for i_atom in aa_residues[i_res].atoms:
                    atom_name = aa_atom_name[i_atom]
                    if atom_name[0] == 'H':
                        continue
                    else:
                        cg_idx.append(i_atom)
                i_bead += 1
                i_resi += 1
                cg_residues.append(CGResidue(i_resi, res_name, "CA", cg_idx[:]))
        elif mol_type == MOL_DNA:
            tmp_atom_index_O3p = 0
            for i_local_index, i_res in enumerate( chain.residues ):
                res_name = aa_residues[i_res].name
                cg_DP_idx = [tmp_atom_index_O3p]
                cg_DS_idx = []
                cg_DB_idx = []
                for i_atom in aa_residues[i_res].atoms:
                    atom_name = aa_atom_name[i_atom]
                    if atom_name[0] == 'H':
                        continue
                    elif atom_name in ATOM_NAME_SET_DP:
                        cg_DP_idx.append(i_atom)
                    elif atom_name in ATOM_NAME_SET_DS:
                        cg_DS_idx.append(i_atom)
                    elif atom_name == "O3'":
                        tmp_atom_index_O3p = i_atom
                    else:
                        cg_DB_idx.append(i_atom)
                i_resi += 1
                if i_local_index > 1:
                    i_bead += 1
                    cg_residues.append(CGResidue(i_resi, res_name, "DP", cg_DP_idx[:]))
                i_bead += 1
                cg_residues.append( CGResidue(i_resi, res_name, "DS", cg_DS_idx[:]))
                i_bead += 1
                cg_residues.append( CGResidue(i_resi, res_name, "DB", cg_DB_idx[:]))
        elif mol_type == MOL_RNA:
            for i_local_index, i_res in enumerate( chain.residues ):
                res_name = aa_residues[i_res].name
                cg_RP_idx = []
                cg_RS_idx = []
                cg_RB_idx = []
                for i_atom in aa_residues[i_res].atoms:
                    atom_name = aa_atom_name[i_atom]
                    if atom_name[0] == 'H':
                        continue
                    elif atom_name in ATOM_NAME_SET_RP:
                        cg_RP_idx.append(i_atom)
                    elif atom_name in ATOM_NAME_SET_RS:
                        cg_RS_idx.append(i_atom)
                    else:
                        cg_RB_idx.append(i_atom)
                i_resi += 1
                if i_local_index > 1:
                    i_bead += 1
                    cg_residues.append( CGResidue(i_resi, res_name, "RP", cg_RP_idx[:]))
                i_bead += 1
                cg_residues.append( CGResidue(i_resi, res_name, "RS", cg_RS_idx[:]))
                i_bead += 1
                cg_residues.append( CGResidue(i_resi, res_name, "RB", cg_RB_idx[:]))
        cg_chains.append(CGChain(i_offset_cg_particle, i_bead - 1, mol_type))
        i_offset_cg_particle += cg_chain_length[i_chain]
        i_offset_cg_residue  += len(chain.residues)

    chain_info_str = "          > Chain {0:>3d} | # particles: {1:>5d} | {2:>5d} -- {3:>5d} "
    for i_chain in range(aa_num_chain):
        print(chain_info_str.format(i_chain,
                                    cg_chain_length[i_chain],
                                    cg_chains[i_chain].first + 1,
                                    cg_chains[i_chain].last  + 1))

    print("------------------------------------------------------------")
    print("          In total: {0} CG particles.".format(cg_num_particles))




    # =========================================================================
    #        ____ ____   _____ ___  ____   ___  _     ___   ______   __
    #       / ___/ ___| |_   _/ _ \|  _ \ / _ \| |   / _ \ / ___\ \ / /
    #      | |  | |  _    | || | | | |_) | | | | |  | | | | |  _ \ V /
    #      | |__| |_| |   | || |_| |  __/| |_| | |__| |_| | |_| | | |
    #       \____\____|   |_| \___/|_|    \___/|_____\___/ \____| |_|
    #
    # =========================================================================

    cg_resid_name  = ["    " for _ in range(cg_num_particles)]
    cg_resid_index = np.zeros(cg_num_particles, dtype=int)
    cg_bead_name   = ["    " for _ in range(cg_num_particles)]
    cg_bead_type   = ["    " for _ in range(cg_num_particles)]
    cg_bead_charge = np.zeros(cg_num_particles)
    cg_bead_mass   = np.zeros(cg_num_particles)
    cg_bead_coor   = np.zeros((cg_num_particles, 3))
    cg_chain_id    = np.zeros(cg_num_particles, dtype=int)

    # protein
    top_cg_pro_bonds         = []
    top_cg_pro_angles        = []
    top_cg_pro_dihedrals     = []
    top_cg_pro_aicg13        = []
    top_cg_pro_aicg14        = []
    top_cg_pro_aicg_contact  = []

    param_cg_pro_e_13        = []
    param_cg_pro_e_14        = []
    param_cg_pro_e_contact   = []

    # DNA
    top_cg_DNA_bonds         = []
    top_cg_DNA_angles        = []
    top_cg_DNA_dih_Gaussian  = []
    top_cg_DNA_dih_periodic  = []

    # RNA
    top_cg_RNA_bonds         = []
    top_cg_RNA_angles        = []
    top_cg_RNA_dihedrals     = []
    top_cg_RNA_base_stack    = []
    top_cg_RNA_base_pair     = []
    top_cg_RNA_other_contact = []

    # protein-DNA
    top_cg_pro_DNA_pwmcos    = []

    # protein-RNA
    top_cg_pro_RNA_contact   = []

    # =================================
    # Step 4: AICG2+ model for proteins
    # =================================
    #                  _       _
    #  _ __  _ __ ___ | |_ ___(_)_ __
    # | '_ \| '__/ _ \| __/ _ \ | '_ \
    # | |_) | | | (_) | ||  __/ | | | |
    # | .__/|_|  \___/ \__\___|_|_| |_|
    # |_|
    #
    # =================================

    if num_chain_pro > 0:
        i_step += 1
        print("============================================================")
        print("> Step {0:>2d}: processing proteins.".format(i_step))

        # --------------------------------
        # Step 4.1: find out C-alpha atoms
        # --------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.1: determine CA mass, charge, and coordinates.".format(i_step))

        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_PROTEIN:
                continue

            for i_res in range( chain.first, chain.last + 1 ):
                res_name = cg_residues[i_res].res_name
                for i_atom in cg_residues[i_res].atoms:
                    if aa_atom_name[i_atom] == "CA":
                        cg_resid_name  [i_res] = res_name
                        cg_resid_index [i_res] = cg_residues     [i_res].res_idx
                        cg_bead_name   [i_res] = "CA"
                        cg_bead_type   [i_res] = res_name
                        cg_bead_charge [i_res] = RES_CHARGE_DICT [res_name]
                        cg_bead_mass   [i_res] = RES_MASS_DICT   [res_name]
                        cg_bead_coor   [i_res] = aa_coor         [i_atom]
                        cg_chain_id    [i_res] = i_chain
                        break

        if len(protein_charge_filename) > 0:
            try:
                with open(protein_charge_filename, 'r') as pro_c_fin:
                    for line in pro_c_fin:
                        charge_data = line.split()
                        if len(charge_data) < 1:
                            continue
                        i = int(charge_data[0])
                        c = float(charge_data[1])
                        cg_bead_charge[i - 1] = c
            except:
                print("ERROR in user-defined charge distribution.\n")
                exit()
        print(">           ... DONE!")

        # -------------------------
        # Step 4.2: AICG2+ topology
        # -------------------------
        print("------------------------------------------------------------")
        print(">      {0}.2: AICG2+ topology.".format(i_step))
        print(" - - - - - - - - - - - - - - - - - - - - - - - -")
        print(">      {0}.2.1: AICG2+ local interactions.".format(i_step))
        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_PROTEIN:
                continue

            for i_res in range( chain.first, chain.last ):
                coor1 = cg_bead_coor[i_res]
                coor2 = cg_bead_coor[i_res + 1]
                dist12 = compute_distance(coor1, coor2)
                top_cg_pro_bonds.append((i_res, dist12))
        print(">           ... Bond: DONE!")

        e_ground_local = 0.0
        e_ground_13    = 0.0
        num_angle      = 0
        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_PROTEIN:
                continue

            for i_res in range(chain.first, chain.last - 1):
                coor1  = cg_bead_coor[i_res     ]
                coor3  = cg_bead_coor[i_res + 2 ]
                dist13 = compute_distance (coor1, coor3)
                top_cg_pro_angles.append  (i_res)
                top_cg_pro_aicg13.append  ( (i_res, dist13))

                # count AICG2+ atomic contact
                contact_counts = count_aicg_atomic_contact(cg_residues   [i_res     ],
                                                           cg_residues   [i_res + 2 ],
                                                           cg_resid_name [i_res     ],
                                                           cg_resid_name [i_res + 2 ],
                                                           aa_atom_name,
                                                           aa_coor)

                # calculate AICG2+ pairwise energy
                e_local = np.dot(AICG_PAIRWISE_ENERGY, contact_counts)
                if e_local > AICG_ENE_UPPER_LIM:
                    e_local = AICG_ENE_UPPER_LIM
                if e_local < AICG_ENE_LOWER_LIM:
                    e_local = AICG_ENE_LOWER_LIM
                e_ground_local += e_local
                e_ground_13    += e_local
                num_angle      += 1
                param_cg_pro_e_13.append( e_local)
        print(">           ... Angle: DONE!")

        e_ground_14 = 0.0
        num_dih = 0
        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_PROTEIN:
                continue

            for i_res in range(chain.first, chain.last - 2):
                coor1 = cg_bead_coor[i_res]
                coor2 = cg_bead_coor[i_res + 1]
                coor3 = cg_bead_coor[i_res + 2]
                coor4 = cg_bead_coor[i_res + 3]
                dihed = compute_dihedral(coor1, coor2, coor3, coor4)
                top_cg_pro_dihedrals.append(i_res)
                top_cg_pro_aicg14.append((i_res, dihed))

                # count AICG2+ atomic contact
                contact_counts = count_aicg_atomic_contact(cg_residues   [i_res     ],
                                                           cg_residues   [i_res + 3 ],
                                                           cg_resid_name [i_res     ],
                                                           cg_resid_name [i_res + 3 ],
                                                           aa_atom_name,
                                                           aa_coor)

                # calculate AICG2+ pairwise energy
                e_local = np.dot(AICG_PAIRWISE_ENERGY, contact_counts)
                if e_local > AICG_ENE_UPPER_LIM:
                    e_local = AICG_ENE_UPPER_LIM
                if e_local < AICG_ENE_LOWER_LIM:
                    e_local = AICG_ENE_LOWER_LIM
                e_ground_local += e_local
                e_ground_14    += e_local
                num_dih      += 1
                param_cg_pro_e_14.append( e_local)
        print(">           ... Dihedral: DONE!")

        # ------------------------
        # Normalize local energies
        # ------------------------
        e_ground_local /= (num_angle + num_dih)
        e_ground_13    /= num_angle
        e_ground_14    /= num_dih

        if scale_scheme == 0:
            for i in range(len(param_cg_pro_e_13)):
                param_cg_pro_e_13[i] *= AICG_13_AVE / e_ground_13
            for i in range(len(param_cg_pro_e_14)):
                param_cg_pro_e_14[i] *= AICG_14_AVE / e_ground_14
        elif scale_scheme == 1:
            for i in range(len(param_cg_pro_e_13)):
                param_cg_pro_e_13[i] *= -AICG_13_GEN
            for i in range(len(param_cg_pro_e_14)):
                param_cg_pro_e_14[i] *= -AICG_14_GEN



        # -----------------------
        # Go type native contacts
        # -----------------------
        print(" - - - - - - - - - - - - - - - - - - - - - - - -")
        print(">      {0}.2.2: AICG2+ Go-type native contacts.".format(i_step))
        e_ground_contact = 0.0
        num_contact      = 0

        # intra-molecular contacts
        print("        Calculating intra-molecular contacts...")
        for i_chain in tqdm( range(aa_num_chain) ):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_PROTEIN:
                continue

            for i_res in range(chain.first, chain.last - 3):
                coor_cai = cg_bead_coor[i_res]
                for j_res in range(i_res + 4, chain.last):
                    coor_caj = cg_bead_coor[j_res]
                    if is_protein_go_contact(cg_residues[i_res], cg_residues[j_res], aa_atom_name, aa_coor):
                        native_dist = compute_distance(coor_cai, coor_caj)
                        num_contact += 1
                        top_cg_pro_aicg_contact.append((i_res, j_res, native_dist))

                        # count AICG2+ atomic contact
                        contact_counts = count_aicg_atomic_contact(cg_residues   [i_res],
                                                                   cg_residues   [j_res],
                                                                   cg_resid_name [i_res],
                                                                   cg_resid_name [j_res],
                                                                   aa_atom_name,
                                                                   aa_coor)

                        # calculate AICG2+ pairwise energy
                        e_local = np.dot(AICG_PAIRWISE_ENERGY, contact_counts)
                        if e_local > AICG_ENE_UPPER_LIM:
                            e_local = AICG_ENE_UPPER_LIM
                        if e_local < AICG_ENE_LOWER_LIM:
                            e_local = AICG_ENE_LOWER_LIM
                        e_ground_contact += e_local
                        num_contact      += 1
                        param_cg_pro_e_contact.append( e_local)
        print(">           ... intra-molecular contacts: DONE!")

        # inter-molecular ( protein-protein ) contacts
        print("        Calculating inter-molecular contacts...")
        for i_chain in tqdm( range(aa_num_chain - 1) ):
            chain1 = cg_chains[i_chain]

            if chain1.moltype != MOL_PROTEIN:
                continue

            for j_chain in range(i_chain + 1, aa_num_chain):
                chain2 = cg_chains[j_chain]

                if chain2.moltype != MOL_PROTEIN:
                    continue

                for i_res in range(chain1.first, chain1.last + 1):
                    coor_cai = cg_bead_coor[i_res]
                    for j_res in range(chain2.first, chain2.last + 1):
                        coor_caj = cg_bead_coor[j_res]
                        if is_protein_go_contact(cg_residues[i_res], cg_residues[j_res], aa_atom_name, aa_coor):
                            native_dist = compute_distance(coor_cai, coor_caj)
                            num_contact += 1
                            top_cg_pro_aicg_contact.append((i_res, j_res, native_dist))

                            # count AICG2+ atomic contact
                            contact_counts = count_aicg_atomic_contact(cg_residues   [i_res],
                                                                       cg_residues   [j_res],
                                                                       cg_resid_name [i_res],
                                                                       cg_resid_name [j_res],
                                                                       aa_atom_name,
                                                                       aa_coor)

                            # calculate AICG2+ pairwise energy
                            e_local = np.dot(AICG_PAIRWISE_ENERGY, contact_counts)
                            if e_local > AICG_ENE_UPPER_LIM:
                                e_local = AICG_ENE_UPPER_LIM
                            if e_local < AICG_ENE_LOWER_LIM:
                                e_local = AICG_ENE_LOWER_LIM
                            e_ground_contact += e_local
                            num_contact      += 1
                            param_cg_pro_e_contact.append( e_local)
        print(">           ... inter-molecular contacts: DONE!")

        # normalize
        e_ground_contact /= num_contact

        if scale_scheme == 0:
            for i in range(len(param_cg_pro_e_contact)):
                param_cg_pro_e_contact[i] *= AICG_CONTACT_AVE / e_ground_contact
        elif scale_scheme == 1:
            for i in range(len(param_cg_pro_e_contact)):
                param_cg_pro_e_contact[i] *= -AICG_CONTACT_GEN

        print("------------------------------------------------------------")
        print("          > Total number of protein contacts: {0:>12d}".format(len( top_cg_pro_aicg_contact )))


    # =============================
    # Step 5: 3SPN.2C model for DNA
    # =============================
    #         _
    #      __| |_ __   __ _
    #     / _` | '_ \ / _` |
    #    | (_| | | | | (_| |
    #     \__,_|_| |_|\__,_|
    #
    # =============================

    if num_chain_DNA > 0:
        i_step += 1
        print("============================================================")
        print("> Step {0:>2d}: processing DNA.".format(i_step))

        # ----------------------------------
        #        Step 5.1: determine P, S, B
        # ----------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.1: determine P, S, B mass, charge, and coordinates.".format(i_step))

        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_DNA:
                continue

            for i_res in range(chain.first, chain.last + 1):
                res_name  = cg_residues[i_res].res_name
                bead_name = cg_residues[i_res].atm_name
                bead_type = bead_name if bead_name == "DP" or bead_name == "DS" else res_name
                bead_coor = compute_center_of_mass(cg_residues[i_res].atoms, aa_atom_name, aa_coor)
                cg_resid_name  [i_res] = res_name
                cg_resid_index [i_res] = cg_residues[i_res].res_idx
                cg_bead_name   [i_res] = bead_name
                cg_bead_type   [i_res] = bead_type
                cg_bead_charge [i_res] = RES_CHARGE_DICT [bead_type]
                cg_bead_mass   [i_res] = RES_MASS_DICT   [bead_type]
                cg_bead_coor   [i_res] = bead_coor
                cg_chain_id    [i_res] = i_chain

        print(">           ... DONE!")

        # ---------------------------------
        #        Step 5.2: 3SPN.2C topology
        # ---------------------------------
        if gen_3spn_itp:
            print("------------------------------------------------------------")
            print(">      {0}.2: 3SPN.2C topology.".format(i_step))
            print(" - - - - - - - - - - - - - - - - - - - - - - - -")
            print(">      {0}.2.1: 3SPN.2C local interactions.".format(i_step))
            print("        Calculating intra-molecular contacts...")
            for i_chain in tqdm( range(aa_num_chain) ):
                chain = cg_chains[i_chain]

                if chain.moltype != MOL_DNA:
                    continue

                for i_res in range(chain.first, chain.last + 1):
                    if cg_bead_name[i_res] == "DS":
                        # bond S--B
                        coor_s = cg_bead_coor[i_res]
                        coor_b = cg_bead_coor[i_res + 1]
                        r_sb   = compute_distance(coor_s, coor_b)
                        top_cg_DNA_bonds.append(( i_res, i_res + 1, r_sb ))
                        if i_res + 3 < chain.last:
                            # bond S--P+1
                            coor_p3 = cg_bead_coor[i_res + 2]
                            r_sp3   = compute_distance(coor_s, coor_p3)
                            top_cg_DNA_bonds.append(( i_res, i_res + 2, r_sp3 ))
                            # Angle S--P+1--S+1
                            resname5  = cg_resid_name [i_res]     [-1]
                            resname3  = cg_resid_name [i_res + 3] [-1]
                            coor_s3   = cg_bead_coor  [i_res + 3]
                            ang_sp3s3 = compute_angle(coor_s, coor_p3, coor_s3)
                            k         = get_DNA3SPN_angle_param("SPS", resname5 + resname3)
                            top_cg_DNA_angles.append(( i_res, i_res + 2, i_res + 3, ang_sp3s3, k * 2 ))
                            # Dihedral S--P+1--S+1--B+1
                            coor_b3     = cg_bead_coor[i_res + 4]
                            dih_sp3s3b3 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_b3)
                            top_cg_DNA_dih_periodic.append(( i_res, i_res + 2, i_res + 3, i_res + 4, dih_sp3s3b3 -180.0))
                            # Dihedral S--P+1--S+1--P+2
                            if i_res + 6 < chain.last:
                                coor_p33     = cg_bead_coor[i_res + 5]
                                dih_sp3s3p33 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_p33)
                                top_cg_DNA_dih_periodic.append(( i_res, i_res + 2, i_res + 3, i_res + 5, dih_sp3s3p33 - 180.0))
                                top_cg_DNA_dih_Gaussian.append(( i_res, i_res + 2, i_res + 3, i_res + 5, dih_sp3s3p33 ))
                    elif cg_bead_name[i_res] == "DP":
                        # bond P--S
                        coor_p = cg_bead_coor[i_res]
                        coor_s = cg_bead_coor[i_res + 1]
                        r_ps   = compute_distance(coor_p, coor_s)
                        top_cg_DNA_bonds.append(( i_res, i_res + 1, r_ps ))
                        # angle P--S--B
                        resname5 = cg_resid_name [i_res - 1] [-1]
                        resname3 = cg_resid_name [i_res + 2] [-1]
                        coor_b   = cg_bead_coor  [i_res + 2]
                        ang_psb  = compute_angle(coor_p, coor_s, coor_b)
                        k        = get_DNA3SPN_angle_param("PSB", resname5 + resname3)
                        top_cg_DNA_angles.append(( i_res, i_res + 1, i_res + 2, ang_psb, k * 2 ))
                        if i_res + 4 < chain.last:
                            # angle P--S--P+1
                            coor_p3  = cg_bead_coor[i_res + 3]
                            ang_psp3 = compute_angle(coor_p, coor_s, coor_p3)
                            k        = get_DNA3SPN_angle_param("PSP", "all")
                            top_cg_DNA_angles.append(( i_res, i_res + 1, i_res + 3, ang_psp3, k * 2 ))
                            # Dihedral P--S--P+1--S+1
                            coor_s3    = cg_bead_coor[i_res + 4]
                            dih_psp3s3 = compute_dihedral(coor_p, coor_s, coor_p3, coor_s3)
                            top_cg_DNA_dih_periodic.append(( i_res, i_res + 1, i_res + 3, i_res + 4, dih_psp3s3 - 180.0))
                            top_cg_DNA_dih_Gaussian.append(( i_res, i_res + 1, i_res + 3, i_res + 4, dih_psp3s3 ))
                    elif cg_bead_name[i_res] == "DB":
                        if i_res + 2 < chain.last:
                            # angle B--S--P+1
                            resname5 = cg_resid_name [i_res]     [-1]
                            resname3 = cg_resid_name [i_res + 1] [-1]
                            coor_b   = cg_bead_coor  [i_res]
                            coor_s   = cg_bead_coor  [i_res - 1]
                            coor_p3  = cg_bead_coor  [i_res + 1]
                            ang_bsp3 = compute_angle(coor_b, coor_s, coor_p3)
                            k        = get_DNA3SPN_angle_param("BSP", resname5 + resname3)
                            top_cg_DNA_angles.append(( i_res, i_res - 1, i_res + 1, ang_bsp3, k * 2 ))
                            # Dihedral B--S--P+1--S+1
                            coor_s3    = cg_bead_coor[i_res + 2]
                            dih_bsp3s3 = compute_dihedral(coor_b, coor_s, coor_p3, coor_s3)
                            top_cg_DNA_dih_periodic.append(( i_res, i_res - 1, i_res + 1, i_res + 2, dih_bsp3s3 - 180.0))
                    else:
                        errmsg = "BUG: Wrong DNA particle type in chain {}, residue {} : {} "
                        print(errmsg.format(i_chain, i_res, res_name))
                        exit()
            print(">           ... Bond, Angle, Dihedral: DONE!")


    # =========================
    # RNA structure based model
    # =========================
    #     ____  _   _    _    
    #    |  _ \| \ | |  / \   
    #    | |_) |  \| | / _ \  
    #    |  _ <| |\  |/ ___ \ 
    #    |_| \_\_| \_/_/   \_\
    # 
    # =========================

    if num_chain_RNA > 0:
        i_step += 1
        print("============================================================")
        print("> Step {0:>2d}: processing RNA.".format(i_step))

        # ----------------------------------
        #         determine P, S, B
        # ----------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.1: determine P, S, B mass, charge, and coordinates.".format(i_step))

        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_RNA:
                continue

            for i_res in range(chain.first, chain.last + 1):
                res_name  = cg_residues[i_res].res_name
                bead_name = cg_residues[i_res].atm_name
                bead_type = bead_name if bead_name == "RP" or bead_name == "RS" else res_name
                cg_resid_name  [i_res] = res_name
                cg_resid_index [i_res] = cg_residues     [i_res].res_idx
                cg_bead_name   [i_res] = bead_name
                cg_bead_type   [i_res] = bead_type
                cg_bead_charge [i_res] = RES_CHARGE_DICT [bead_type]
                cg_bead_mass   [i_res] = RES_MASS_DICT   [bead_type]
                cg_chain_id    [i_res] = i_chain
                if bead_name == "RP":
                    for i_atom in cg_residues[i_res].atoms:
                        if aa_atom_name[i_atom][1] == 'P':
                            bead_coor = aa_coor[i_atom]
                elif bead_name == "RS":
                    total_mass = 0
                    tmp_coor   = np.zeros(3)
                    for i_atom in cg_residues[i_res].atoms:
                        a_name = aa_atom_name[i_atom]
                        if a_name in ["C1'", "C2'", "C3'", "C4'", "O4'"]:
                            a_mass      = ATOM_MASS_DICT[a_name[0]]
                            a_coor      = aa_coor[i_atom]
                            total_mass += a_mass
                            tmp_coor   += a_coor * a_mass
                    bead_coor = tmp_coor / total_mass
                elif bead_name == "RB":
                    if res_name[-1] == 'A' or res_name[-1] == 'G':
                        for i_atom in cg_residues[i_res].atoms:
                            if aa_atom_name[i_atom] == "N1":
                                bead_coor = aa_coor[i_atom]
                    else:
                        for i_atom in cg_residues[i_res].atoms:
                            if aa_atom_name[i_atom] == "N3":
                                bead_coor = aa_coor[i_atom]
                cg_bead_coor[i_res] = bead_coor

        print(">           ... DONE!")

        # -------------------------
        # Step 6.2: RNA topology
        # -------------------------
        print("------------------------------------------------------------")
        print(">      {0}.2: RNA topology.".format(i_step))
        print(" - - - - - - - - - - - - - - - - - - - - - - - -")
        print(">      {0}.2.1: RNA local interactions.".format(i_step))

        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_RNA:
                continue

            print("        Calculating intra-molecular contacts...")
            for i_res in tqdm( range(chain.first, chain.last + 1) ):
                if cg_bead_name[i_res] == "RS":
                    # bond S--B
                    coor_s    = cg_bead_coor[i_res]
                    coor_b    = cg_bead_coor[i_res + 1]
                    r_sb      = compute_distance(coor_s, coor_b)
                    base_type = "R" if cg_resid_name[i_res] in ["RA", "RG"] else "Y"
                    bond_type = "S" + base_type
                    k         = RNA_BOND_K_LIST[bond_type] * CAL2JOU
                    top_cg_RNA_bonds.append((i_res, i_res + 1, r_sb , k * 2 * 100.0))
                    # bond S--P+1
                    if i_res + 2 < chain.last:
                        coor_p3 = cg_bead_coor[i_res + 2]
                        r_sp3   = compute_distance(coor_s, coor_p3)
                        k       = RNA_BOND_K_LIST["SP"] * CAL2JOU
                        top_cg_RNA_bonds.append((i_res, i_res + 2, r_sp3 , k * 2 * 100.0))
                    if i_res + 4 <= chain.last:
                        # Angle S--P+1--S+1
                        coor_s3   = cg_bead_coor[i_res + 3]
                        ang_sp3s3 = compute_angle(coor_s, coor_p3, coor_s3)
                        k         = RNA_ANGLE_K_LIST["SPS"] * CAL2JOU
                        top_cg_RNA_angles.append((i_res, i_res + 2, i_res + 3, ang_sp3s3, k * 2))
                        # Dihedral S--P+1--S+1--B+1
                        coor_b3     = cg_bead_coor[i_res + 4]
                        dih_sp3s3b3 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_b3)
                        base_type   = "R" if cg_resid_name[i_res + 4] in ["RA", "RG"] else "Y"
                        dihe_type   = "SPS" + base_type
                        k           = RNA_DIHEDRAL_K_LIST[dihe_type] * CAL2JOU
                        top_cg_RNA_dihedrals.append((i_res, i_res + 2, i_res + 3, i_res + 4, dih_sp3s3b3, k))
                    # Dihedral S--P+1--S+1--P+2
                    if i_res + 5 < chain.last:
                        coor_p33     = cg_bead_coor[i_res + 5]
                        dih_sp3s3p33 = compute_dihedral(coor_s, coor_p3, coor_s3, coor_p33)
                        k            = RNA_DIHEDRAL_K_LIST["SPSP"] * CAL2JOU
                        top_cg_RNA_dihedrals.append((i_res, i_res + 2, i_res + 3, i_res + 5, dih_sp3s3p33, k))
                elif cg_bead_name[i_res] == "RP":
                    # bond P--S
                    coor_p = cg_bead_coor[i_res]
                    coor_s = cg_bead_coor[i_res + 1]
                    r_ps   = compute_distance(coor_p, coor_s)
                    k      = RNA_BOND_K_LIST["PS"] * CAL2JOU
                    top_cg_RNA_bonds.append((i_res, i_res + 1, r_ps , k * 2 * 100.0))
                    # angle P--S--B
                    coor_b    = cg_bead_coor[i_res + 2]
                    ang_psb   = compute_angle(coor_p, coor_s, coor_b)
                    base_type = "R" if cg_resid_name[i_res + 2] in ["RA", "RG"] else "Y"
                    angl_type = "PS" + base_type
                    k         = RNA_ANGLE_K_LIST[angl_type] * CAL2JOU
                    top_cg_RNA_angles.append((i_res, i_res + 1, i_res + 2, ang_psb, k * 2))
                    if i_res + 4 < chain.last:
                        # angle P--S--P+1
                        coor_p3  = cg_bead_coor[i_res + 3]
                        ang_psp3 = compute_angle(coor_p, coor_s, coor_p3)
                        k        = RNA_ANGLE_K_LIST["PSP"] * CAL2JOU
                        top_cg_RNA_angles.append((i_res, i_res + 1, i_res + 3, ang_psp3, k * 2))
                        # Dihedral P--S--P+1--S+1
                        coor_s3    = cg_bead_coor[i_res + 4]
                        dih_psp3s3 = compute_dihedral(coor_p, coor_s, coor_p3, coor_s3)
                        k          = RNA_DIHEDRAL_K_LIST["PSPS"] * CAL2JOU
                        top_cg_RNA_dihedrals.append((i_res, i_res + 1, i_res + 3, i_res + 4, dih_psp3s3, k))
                elif cg_bead_name[i_res] == "RB":
                    # do nothing...
                    pass

        # -----------------------
        # Go type native contacts
        # -----------------------
        print(" - - - - - - - - - - - - - - - - - - - - - - - -")
        print(">      {0}.2.2: RNA Go-type native contacts.".format(i_step))
        print( "        Calculating intra-molecular contacts..." )
        for i_chain in range(aa_num_chain):
            chain = cg_chains[i_chain]

            if chain.moltype != MOL_RNA:
                continue

            for i_res in range(chain.first, chain.last - 2):

                if cg_bead_name[i_res] == "RP":
                    continue

                coor_i = cg_bead_coor[i_res]

                for j_res in range(i_res + 3, chain.last + 1):

                    if cg_bead_name[j_res] == "RP":
                        continue

                    if cg_bead_name[i_res] == "RS" or cg_bead_name[j_res] == "RS":
                        if j_res < i_res + 6:
                            continue

                    coor_j = cg_bead_coor[j_res]
                    native_dist = compute_distance(coor_i, coor_j)
                    adist, nhb  = compute_RNA_Go_contact(cg_residues[i_res],
                                                         cg_residues[j_res],
                                                         aa_atom_name,
                                                         aa_coor)

                    if adist > RNA_GO_ATOMIC_CUTOFF:
                        continue
                    
                    if j_res == i_res + 3 and cg_bead_name[i_res] == "RB":
                        coor_i_sug = cg_bead_coor[i_res - 1]
                        coor_j_sug = cg_bead_coor[j_res - 1]
                        st_dih = compute_dihedral(coor_i, coor_i_sug, coor_j_sug, coor_j)
                        if abs( st_dih ) < RNA_STACK_DIH_CUTOFF and adist < RNA_STACK_DIST_CUTOFF:
                            top_cg_RNA_base_stack.append((i_res, j_res, native_dist, RNA_STACK_EPSILON))
                        else:
                            top_cg_RNA_other_contact.append((i_res, j_res, native_dist, RNA_PAIR_EPSILON_OTHER["BB"]))
                    elif cg_bead_name[i_res] == "RB" and cg_bead_name[j_res] == "RB":
                        if nhb == 2:
                            top_cg_RNA_base_pair.append((i_res, j_res, native_dist, RNA_BPAIR_EPSILON_2HB))
                        elif nhb >= 3:
                            top_cg_RNA_base_pair.append((i_res, j_res, native_dist, RNA_BPAIR_EPSILON_3HB))
                        else:
                            top_cg_RNA_other_contact.append((i_res, j_res, native_dist, RNA_PAIR_EPSILON_OTHER["BB"]))
                    else:
                        contact_type = cg_bead_name[i_res][-1] * cg_bead_name[j_res][-1]
                        top_cg_RNA_other_contact.append((i_res, j_res, native_dist, RNA_PAIR_EPSILON_OTHER[contact_type]))
 
        print( "        Calculating inter-molecular contacts..." )
        for i_chain in tqdm( range(aa_num_chain) ):
            chain_1 = cg_chains[i_chain]
            if chain_1.moltype != MOL_RNA:
                continue
            for i_res in range(chain_1.first, chain_1.last + 1):
                if cg_bead_name[i_res] == "RP":
                    continue
                coor_i = cg_bead_coor[i_res]
                for j_chain in range(i_chain + 1, aa_num_chain):
                    chain_2 = cg_chains[j_chain]
                    if chain_2.moltype != MOL_RNA:
                        continue
                    for j_res in range(chain_2.first, chain_2.last + 1):
                        if cg_bead_name[j_res] == "RP":
                            continue
                        coor_j = cg_bead_coor[j_res]
                        native_dist = compute_distance(coor_i, coor_j)
                        adist, nhb  = compute_RNA_Go_contact(cg_residues[i_res],
                                                             cg_residues[j_res],
                                                             aa_atom_name,
                                                             aa_coor)
                        if adist > RNA_GO_ATOMIC_CUTOFF:
                            continue
                        if cg_bead_name[i_res] == "RB" and cg_bead_name[j_res] == "RB":
                            if nhb == 2:
                                top_cg_RNA_base_pair.append((i_res, j_res, native_dist, RNA_BPAIR_EPSILON_2HB))
                            elif nhb >= 3:
                                top_cg_RNA_base_pair.append((i_res, j_res, native_dist, RNA_BPAIR_EPSILON_3HB))
                            else:
                                top_cg_RNA_other_contact.append((i_res, j_res, native_dist, RNA_PAIR_EPSILON_OTHER["BB"]))
                        else:
                            contact_type = cg_bead_name[i_res][-1] * cg_bead_name[j_res][-1]
                            top_cg_RNA_other_contact.append((i_res, j_res, native_dist, RNA_PAIR_EPSILON_OTHER[contact_type]))
 
        print(">           ... DONE!")
        print("------------------------------------------------------------")
        num_rna_contacts = len(top_cg_RNA_base_stack) + len(top_cg_RNA_base_pair) + len(top_cg_RNA_other_contact) 
        print("          > Total number of RNA contacts:     {0:>12d}".format(num_rna_contacts))



    # ===========================================================
    # Protein-RNA structure-based interactions: Go-like potential
    # ===========================================================
    #                  _       _             ____  _   _    _    
    #  _ __  _ __ ___ | |_ ___(_)_ __       |  _ \| \ | |  / \   
    # | '_ \| '__/ _ \| __/ _ \ | '_ \ _____| |_) |  \| | / _ \  
    # | |_) | | | (_) | or  __/ | | | |_____|  _ <| |\  |/ ___ \ 
    # | .__/|_|  \___/ \__\___|_|_| |_|     |_| \_\_| \_/_/   \_\
    # |_|                                                        
    # 
    # ============================================================

    if num_chain_RNA > 0 and num_chain_pro > 0:
        i_step += 1
        print("============================================================")
        print("> Step {0:>2d}: Generating protein-RNA native contacts.".format(i_step))

        print("        Calculating protein-RNA contacts...")
        for i_chain in tqdm( range(aa_num_chain) ):
            chain_pro = cg_chains[i_chain]
            if chain_pro.moltype != MOL_PROTEIN:
                continue
            for i_res in range(chain_pro.first, chain_pro.last + 1):
                coor_i = cg_bead_coor[i_res]

                for j_chain in range(1, aa_num_chain + 1):
                    chain_RNA = cg_chains[j_chain]
                    if chain_RNA.moltype != MOL_RNA:
                        continue
                    for j_res in range(chain_RNA.first, chain_RNA.last + 1):
                        if cg_bead_name[j_res] == "RP":
                            continue
                        if not is_protein_RNA_go_contact(cg_residues[i_res], cg_residues[j_res], aa_atom_name, aa_coor):
                            continue
                        coor_j = cg_bead_coor[j_res]
                        native_dist = compute_distance(coor_i, coor_j)
                        if cg_bead_name[j_res] == "RS":
                            top_cg_pro_RNA_contact.append((i_res, j_res, native_dist, PRO_RNA_GO_EPSILON_S))
                        elif cg_bead_name[j_res] == "RB":
                            top_cg_pro_RNA_contact.append((i_res, j_res, native_dist, PRO_RNA_GO_EPSILON_B))

        print(">           ... DONE!")
        print("------------------------------------------------------------")
        print("          > Total number of protein-RNA contacts: {0:>8d}  \n".format( len(top_cg_pro_RNA_contact)))


    # ============================================================
    # PWMcos parameters: protein-DNA sequence-specific interaction
    # ============================================================
    #        ______        ____  __               
    #       |  _ \ \      / /  \/  | ___ ___  ___ 
    #       | |_) \ \ /\ / /| |\/| |/ __/ _ \/ __|
    #       |  __/ \ V  V / | |  | | (_| (_) \__ \
    #       |_|     \_/\_/  |_|  |_|\___\___/|___/
    # 
    # ============================================================
    if gen_pwmcos_itp:
        pwmcos_native_contacts = []

        if num_chain_pro == 0:
            error("Cannot generate PWMcos parameters without protein...")
        if num_chain_DNA != 2:
            error("Cannot generate PWMcos parameters from more or less than two DNA chains...")

        i_step += 1
        print("============================================================")
        print("> Step {0:>2d}: Generating PWMcos parameters.".format(i_step))

        # ----------------------------------
        #        Step 7.1: determine P, S, B
        # ----------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.1: determine contacts between protein and DNA.".format(i_step))

        i_count_DNA = 0
        for i_chain in range(aa_num_chain):
            chain_pro = cg_chains[i_chain]

            if chain_pro.moltype != MOL_PROTEIN:
                continue

            for i_res in range(chain_pro.first, chain_pro.last + 1):
                i_res_N = i_res if i_res == chain_pro.first else i_res - 1
                i_res_C = i_res if i_res == chain_pro.last  else i_res + 1

                coor_pro_i = cg_bead_coor [ i_res   ]
                coor_pro_N = cg_bead_coor [ i_res_N ]
                coor_pro_C = cg_bead_coor [ i_res_C ]

                for j_chain in range(aa_num_chain):
                    chain_DNA = cg_chains[j_chain]
                    
                    if chain_DNA.moltype != MOL_DNA:
                        continue

                    for j_res in range(chain_DNA.first + 3, chain_DNA.last - 2):
                        if cg_bead_name[j_res] != "DB":
                            continue
                        if not is_PWMcos_contact(cg_residues[i_res], cg_residues[j_res], aa_atom_name, aa_coor):
                            continue

                        j_res_5, j_res_3 = j_res - 3, j_res + 3
                        coor_dna_j       = cg_bead_coor[ j_res     ]
                        coor_dna_5       = cg_bead_coor[ j_res_5   ]
                        coor_dna_3       = cg_bead_coor[ j_res_3   ]
                        coor_dna_S       = cg_bead_coor[ j_res - 1 ]

                        vec0   = coor_pro_i - coor_dna_j
                        vec1   = coor_dna_S - coor_dna_j
                        vec2   = coor_dna_3 - coor_dna_5
                        vec3   = coor_pro_N - coor_pro_C
                        r0     = np.norm(vec0)
                        theta1 = compute_vec_angle(vec0, vec1)
                        theta2 = compute_vec_angle(vec0, vec2)
                        theta3 = compute_vec_angle(vec0, vec3)

                        push!(pwmcos_native_contacts, (i_res - chain_pro.first + 1,
                                                       cg_resid_index[j_res],
                                                       r0,
                                                       theta1,
                                                       theta2,
                                                       theta3))

                        if do_debug
                            print("PWMcos | pro ===> ", i_res - chain_pro.first + 1,
                                    " DNA ===> ", j_res, " : ", cg_resid_index[j_res],
                                    " r0 = ", r0,
                                    " theta1 = ", theta1,
                                    " theta2 = ", theta2,
                                    " theta3 = ", theta3)

        # ------------------------------------------------
        #        Step 7.2: Read in PFM and convert to PWM
        # ------------------------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.2: read in position frequency matrix (PFM).".format(i_step))

        pwmcos_pwm, pwmcos_chain_a, pwmcos_chain_b = read_modified_pfm(pfm_filename)
        num_pwmcos_terms = len(pwmcos_chain_a)


        # ------------------------------------------------
        #        Step 7.2: Read in PFM and convert to PWM
        # ------------------------------------------------
        print("------------------------------------------------------------")
        print(">      {0}.3: decomposing PWM.".format(i_step))

        ip_count = np.zeros(num_pwmcos_terms)

        contact_to_pwm = []
        for nat_contact in pwmcos_native_contacts:
            i_dna = nat_contact[1] # cg_resid_index[dna]

            if i_dna in pwmcos_chain_a:
                cnt_pwm_idx_a = pwmcos_chain_a.index(i_dna)
                contact_to_pwm.append((cnt_pwm_idx_a, 1) )
                ip_count[cnt_pwm_idx_a] += 1
            elif i_dna in pwmcos_chain_b:
                cnt_pwm_idx_b = pwmcos_chain_b.index(i_dna)
                contact_to_pwm.append((cnt_pwm_idx_b, -1) )
                ip_count[cnt_pwm_idx_b] += 1
            else:
                print("Index error in CHAIN_A or CHAIN_B!")

        pwm_decomposed = pwmcos_pwm / ip_count

        for i_cnt, nat_cnt in enumerate(pwmcos_native_contacts):
            pwm_i, pwm_order = contact_to_pwm[i_cnt][0], contact_to_pwm[i_cnt][1]
            eA, eC, eG, eT = pwm_decomposed[pwm_i, ::pwm_order]
            top_cg_pro_DNA_pwmcos.append((nat_cnt[1],
                                          nat_cnt[3],
                                          nat_cnt[4],
                                          nat_cnt[5],
                                          nat_cnt[6],
                                          eA, eC, eG, eT))

        if do_debug:
            print(size( contact_to_pwm ))
            print(pwm_decomposed)

        print(">           ... DONE!")



    # =========================================================================
    #   ___  _   _ _____ ____  _   _ _____ 
    #  / _ \| | | |_   _|  _ \| | | |_   _|
    # | | | | | | | | | | |_) | | | | | |  
    # | |_| | |_| | | | |  __/| |_| | | |  
    #  \___/ \___/  |_| |_|    \___/  |_|  
    # 
    # =========================================================================

    if gen_pwmcos_itp
        do_output_top    = False
        do_output_itp    = False
        do_output_gro    = False
        do_output_pwmcos = True
    else:
        do_output_top    = True
        do_output_itp    = True
        do_output_gro    = True
        do_output_pwmcos = False
    end

    if do_output_sequence
        do_output_top    = False
        do_output_itp    = False
        do_output_gro    = False
        do_output_pwmcos = False
    end


    i_step += 1
    print("============================================================")
    print("> Step {0:>2d}: output .itp and .gro files.".format(i_step))

    # -------------------------------------------------------------------
    #        top ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # -------------------------------------------------------------------
    if do_output_top
        # ---------------
        #        filename
        # ---------------
        top_name = pdb_name[1:end-4] * "_cg.top"
        top_file = open(top_name, "w")

        itp_name = pdb_name[1:end-4] * "_cg.itp"
        itp_system_name = pdb_name[1:end-4]

        write(top_file, "; atom types for coarse-grained models\n")
        write(top_file, "#include \"./lib/atom_types.itp\" \n")
        if num_chain_pro > 0
            write(top_file, "; AICG2+ flexible local angle parameters \n")
            write(top_file, "#include \"./lib/flexible_local_angle.itp\" \n")
            write(top_file, "; AICG2+ flexible local dihedral parameters \n")
            write(top_file, "#include \"./lib/flexible_local_dihedral.itp\" \n")
        end
        write(top_file, "\n")

        write(top_file, "; Molecule topology \n")
        @printf(top_file, "#include \"./top/%s\" \n\n", itp_name)

        write(top_file, "[ system ] \n")
        @printf(top_file, "%s \n\n", itp_system_name)

        write(top_file, "[ molecules ] \n")
        @printf(top_file, "%s  1 \n\n", itp_system_name)

        write(top_file, "; [ cg_ele_mol_pairs ] \n")
        @printf(top_file, "; ON 1 - 2 : 3 - 4 \n")
        @printf(top_file, "; OFF 1 - 1 : 3 - 3 \n\n")

        write(top_file, "; [ pwmcos_mol_pairs ] \n")
        @printf(top_file, "; ON 1 - 2 : 3 - 4 \n")
        @printf(top_file, "; OFF 1 - 1 : 3 - 3 \n\n")
    end


    # -------------------------------------------------------------------
    #        itp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # -------------------------------------------------------------------
    if do_output_itp
        itp_mol_head     = "[ moleculetype ]\n"
        itp_mol_comm     = format(";{1:15s} {2:6s}\n", "name", "nrexcl")
        itp_mol_line     = "{1:<16} {2:>6d}\n"

        itp_atm_head     = "[ atoms ]\n"
        itp_atm_comm     = format(";{:>9}{:>5}{:>10}{:>5}{:>5}{:>5} {:>8} {:>8}\n", "nr", "type", "resnr", "res", "atom", "cg", "charge", "mass")
        itp_atm_line     = "{:>10d}{:>5}{:>10d}{:>5}{:>5}{:>5d} {:>8.3f} {:>8.3f}\n"

        itp_bnd_head     = "[ bonds ]\n"
        itp_bnd_comm     = format(";{:>9}{:>10}{:>5}{:>18}{:>18}\n", "i", "j", "f", "eq", "coef")
        itp_bnd_line     = "{:>10d}{:>10d}{:>5d}{:>18.4E}{:>18.4E}\n"

        itp_13_head      = "[ angles ] ; AICG2+ 1-3 interaction\n"
        itp_13_comm      = format(";{:>9}{:>10}{:>10}{:>5}{:>15}{:>15}{:>15}\n", "i", "j", "k", "f", "eq", "coef", "w")
        itp_13_line      = "{:>10d}{:>10d}{:>10d}{:>5d}{:>15.4E}{:>15.4E}{:>15.4E}\n"

        itp_ang_f_head   = "[ angles ] ; AICG2+ flexible local interaction\n"
        itp_ang_f_comm   = format(";{:>9}{:>10}{:>10}{:>5}\n", "i", "j", "k", "f")
        itp_ang_f_line   = "{:>10d}{:>10d}{:>10d}{:>5d}\n"

        itp_ang_head     = "[ angles ] ; \n"
        itp_ang_comm     = format(";{:>9}{:>10}{:>10}{:>5}{:>18}{:>18} \n", "i", "j", "k", "f", "eq", "coef")
        itp_ang_line     = "{:>10d}{:>10d}{:>10d}{:>5d}{:>18.4E}{:>18.4E}\n"

        itp_dih_P_head   = "[ dihedrals ] ; periodic dihedrals\n"
        itp_dih_P_comm   = format(";{:>9}{:>10}{:>10}{:>10}{:>5}{:>18}{:>18}{:>5}\n", "i", "j", "k", "l", "f", "eq", "coef", "n")
        itp_dih_P_line   = "{:>10d}{:>10d}{:>10d}{:>10d}{:>5d}{:>18.4E}{:>18.4E}{:>5d}\n"

        itp_dih_G_head   = "[ dihedrals ] ; Gaussian dihedrals\n"
        itp_dih_G_comm   = format(";{:>9}{:>10}{:>10}{:>10}{:>5}{:>15}{:>15}{:>15}\n", "i", "j", "k", "l", "f", "eq", "coef", "w")
        itp_dih_G_line   = "{:>10d}{:>10d}{:>10d}{:>10d}{:>5d}{:>15.4E}{:>15.4E}{:>15.4E}\n"

        itp_dih_F_head   = "[ dihedrals ] ; AICG2+ flexible local interation\n"
        itp_dih_F_comm   = format(";{:>9}{:>10}{:>10}{:>10}{:>5}\n", "i", "j", "k", "l", "f")
        itp_dih_F_line   = "{:>10d}{:>10d}{:>10d}{:>10d}{:>5d}\n"

        itp_contact_head = "[ pairs ] ; Go-type native contact\n"
        itp_contact_comm = format(";{:>9}{:>10}{:>10}{:>15}{:>15}\n", "i", "j", "f", "eq", "coef")
        itp_contact_line = "{:>10d}{:>10d}{:>10d}{:>15.4E}{:>15.4E}\n"

        itp_exc_head     = "[ exclusions ] ; Genesis exclusion list\n"
        itp_exc_comm     = format(";{:>9}{:>10}\n", "i", "j")
        itp_exc_line     = "{:>10d}{:>10d}\n"

        # ---------------
        #        filename
        # ---------------
        itp_name = pdb_name[1:end-4] * "_cg.itp"
        itp_file = open(itp_name, "w")

        # --------------------
        # Writing CG particles
        # --------------------
        # write molecule type information
        itp_system_name = pdb_name[1:end-4]
        write(itp_file, itp_mol_head)
        write(itp_file, itp_mol_comm)
        printfmt(itp_file, itp_mol_line, itp_system_name, MOL_NR_EXCL)
        write(itp_file,"\n")

        # -----------------------
        #               [ atoms ]
        # -----------------------

        write(itp_file, itp_atm_head)
        write(itp_file, itp_atm_comm)
        for i_bead in range(1, cg_num_particles + 1):
            printfmt(itp_file,
                     itp_atm_line,
                     i_bead,
                     cg_bead_type[i_bead],
                     cg_resid_index[i_bead],
                     cg_resid_name[i_bead],
                     cg_bead_name[i_bead],
                     AICG_ATOM_FUNC_NR,
                     cg_bead_charge[i_bead],
                     cg_bead_mass[i_bead])
        end
        write(itp_file,"\n")

        # ----------------
        #        [ bonds ]
        # ----------------

        if len(top_cg_pro_bonds) + len(top_cg_DNA_bonds) + len(top_cg_RNA_bonds) > 0
            write(itp_file, itp_bnd_head)
            write(itp_file, itp_bnd_comm)

            # AICG2+ bonds
            for i_bond in range(1, len(top_cg_pro_bonds) + 1):
                printfmt(itp_file,
                         itp_bnd_line,
                         top_cg_pro_bonds[i_bond][1],
                         top_cg_pro_bonds[i_bond][1] + 1,
                         AICG_BOND_FUNC_TYPE,
                         top_cg_pro_bonds[i_bond][2] * 0.1,
                         AICG_BOND_K)
            end

            # 3SPN.2C bonds
            for i_bond in range(1, len(top_cg_DNA_bonds) + 1):
                printfmt(itp_file,
                         itp_bnd_line,
                         top_cg_DNA_bonds[i_bond][1],
                         top_cg_DNA_bonds[i_bond][2],
                         DNA3SPN_BOND_FUNC4_TYPE,
                         top_cg_DNA_bonds[i_bond][3] * 0.1,
                         DNA3SPN_BOND_K_2)
            end

            # Structure-based RNA bonds
            for i_bond in range(1, len(top_cg_RNA_bonds) + 1):
                printfmt(itp_file,
                         itp_bnd_line,
                         top_cg_RNA_bonds[i_bond][1],
                         top_cg_RNA_bonds[i_bond][2],
                         RNA_BOND_FUNC_TYPE,
                         top_cg_RNA_bonds[i_bond][3] * 0.1,
                         top_cg_RNA_bonds[i_bond][4])
            end
            write(itp_file, "\n")

        end

        # -----------------
        #        [ angles ]
        # -----------------

        # AICG2+ 1-3
        if len(top_cg_pro_aicg13) > 0
            write(itp_file, itp_13_head)
            write(itp_file, itp_13_comm)
            for i_13 in range(1, len(top_cg_pro_aicg13) + 1):
                printfmt(itp_file,
                         itp_13_line,
                         top_cg_pro_aicg13[i_13][1],
                         top_cg_pro_aicg13[i_13][1] + 1,
                         top_cg_pro_aicg13[i_13][1] + 2,
                         AICG_ANG_G_FUNC_TYPE,
                         top_cg_pro_aicg13[i_13][2] * 0.1,
                         param_cg_pro_e_13[i_13] * CAL2JOU,
                         AICG_13_SIGMA)
            end
            write(itp_file, "\n")
        end

        # AICG2+ flexible
        if len(top_cg_pro_angles) > 0
            write(itp_file, itp_ang_f_head)
            write(itp_file, itp_ang_f_comm)
            for i_ang in range(1, len(top_cg_pro_angles) + 1):
                printfmt(itp_file,
                         itp_ang_f_line,
                         top_cg_pro_angles[i_ang],
                         top_cg_pro_angles[i_ang] + 1,
                         top_cg_pro_angles[i_ang] + 2,
                         AICG_ANG_F_FUNC_TYPE)
            end
            write(itp_file, "\n")
        end

        # 3SPN.2C angles
        if len(top_cg_DNA_angles) > 0
            write(itp_file, itp_ang_head)
            write(itp_file, itp_ang_comm)
            for i_ang in range(1, len(top_cg_DNA_angles) + 1):
                printfmt(itp_file,
                         itp_ang_line,
                         top_cg_DNA_angles[i_ang][1],
                         top_cg_DNA_angles[i_ang][2],
                         top_cg_DNA_angles[i_ang][3],
                         DNA3SPN_ANG_FUNC_TYPE,
                         top_cg_DNA_angles[i_ang][4],
                         top_cg_DNA_angles[i_ang][5])
            end
            write(itp_file, "\n")
        end

        # RNA structure-based angles
        if len(top_cg_RNA_angles) > 0
            write(itp_file, itp_ang_head)
            write(itp_file, itp_ang_comm)
            for i_ang in range(1, len(top_cg_RNA_angles) + 1):
                printfmt(itp_file,
                         itp_ang_line,
                         top_cg_RNA_angles[i_ang][1],
                         top_cg_RNA_angles[i_ang][2],
                         top_cg_RNA_angles[i_ang][3],
                         RNA_ANG_FUNC_TYPE,
                         top_cg_RNA_angles[i_ang][4],
                         top_cg_RNA_angles[i_ang][5])
            end
            write(itp_file, "\n")
        end


        # --------------------
        #        [ dihedrals ]
        # --------------------

        # AICG2+ Gaussian dihedrals
        if len(top_cg_pro_aicg14) > 0
            write(itp_file, itp_dih_G_head)
            write(itp_file, itp_dih_G_comm)
            for i_dih in range(1, len(top_cg_pro_aicg14) + 1):
                printfmt(itp_file,
                         itp_dih_G_line,
                         top_cg_pro_aicg14[i_dih][1],
                         top_cg_pro_aicg14[i_dih][1] + 1,
                         top_cg_pro_aicg14[i_dih][1] + 2,
                         top_cg_pro_aicg14[i_dih][1] + 3,
                         AICG_DIH_G_FUNC_TYPE,
                         top_cg_pro_aicg14[i_dih][2],
                         param_cg_pro_e_14[i_dih] * CAL2JOU,
                         AICG_14_SIGMA)
            end
            write(itp_file, "\n")
        end

        # AICG2+ flexible dihedrals
        if len(top_cg_pro_dihedrals) > 0
            write(itp_file, itp_dih_F_head)
            write(itp_file, itp_dih_F_comm)
            for i_dih in range(1, len(top_cg_pro_dihedrals) + 1):
                printfmt(itp_file,
                         itp_dih_F_line,
                         top_cg_pro_dihedrals[i_dih],
                         top_cg_pro_dihedrals[i_dih] + 1,
                         top_cg_pro_dihedrals[i_dih] + 2,
                         top_cg_pro_dihedrals[i_dih] + 3,
                         AICG_DIH_F_FUNC_TYPE)
            end
            write(itp_file, "\n")
        end

        # 3SPN.2C Gaussian dihedrals
        if len(top_cg_DNA_dih_Gaussian) > 0
            write(itp_file, itp_dih_G_head)
            write(itp_file, itp_dih_G_comm)
            for i_dih in range(1, len(top_cg_DNA_dih_Gaussian) + 1):
                printfmt(itp_file,
                         itp_dih_G_line,
                         top_cg_DNA_dih_Gaussian[i_dih][1],
                         top_cg_DNA_dih_Gaussian[i_dih][2],
                         top_cg_DNA_dih_Gaussian[i_dih][3],
                         top_cg_DNA_dih_Gaussian[i_dih][4],
                         DNA3SPN_DIH_G_FUNC_TYPE,
                         top_cg_DNA_dih_Gaussian[i_dih][5],
                         DNA3SPN_DIH_G_K,
                         DNA3SPN_DIH_G_SIGMA)
            end
            write(itp_file, "\n")
        end

        # 3SPN.2C Periodic dihedrals
        if len(top_cg_DNA_dih_periodic) > 0
            write(itp_file, itp_dih_P_head)
            write(itp_file, itp_dih_P_comm)
            for i_dih in range(1, len(top_cg_DNA_dih_periodic) + 1):
                printfmt(itp_file,
                         itp_dih_P_line,
                         top_cg_DNA_dih_periodic[i_dih][1],
                         top_cg_DNA_dih_periodic[i_dih][2],
                         top_cg_DNA_dih_periodic[i_dih][3],
                         top_cg_DNA_dih_periodic[i_dih][4],
                         DNA3SPN_DIH_P_FUNC_TYPE,
                         top_cg_DNA_dih_periodic[i_dih][5],
                         DNA3SPN_DIH_P_K,
                         DNA3SPN_DIH_P_FUNC_PERI)
            end
            write(itp_file, "\n")
        end

        # RNA structure-based Periodic dihedrals
        if len(top_cg_RNA_dihedrals) > 0
            write(itp_file, itp_dih_P_head)
            write(itp_file, itp_dih_P_comm)
            for i_dih in range(1, len(top_cg_RNA_dihedrals) + 1):
                printfmt(itp_file,
                         itp_dih_P_line,
                         top_cg_RNA_dihedrals[i_dih][1],
                         top_cg_RNA_dihedrals[i_dih][2],
                         top_cg_RNA_dihedrals[i_dih][3],
                         top_cg_RNA_dihedrals[i_dih][4],
                         RNA_DIH_FUNC_TYPE,
                         top_cg_RNA_dihedrals[i_dih][5] - 180,
                         top_cg_RNA_dihedrals[i_dih][6],
                         1)
            end
            for i_dih in range(1, len(top_cg_RNA_dihedrals) + 1):
                printfmt(itp_file,
                         itp_dih_P_line,
                         top_cg_RNA_dihedrals[i_dih][1],
                         top_cg_RNA_dihedrals[i_dih][2],
                         top_cg_RNA_dihedrals[i_dih][3],
                         top_cg_RNA_dihedrals[i_dih][4],
                         RNA_DIH_FUNC_TYPE,
                         3 * top_cg_RNA_dihedrals[i_dih][5] - 180,
                         top_cg_RNA_dihedrals[i_dih][6] / 2,
                         3)
            end
            write(itp_file, "\n")
        end


        # ----------------
        #        [ pairs ]
        # ----------------

        # write protein Go-type native contacts
        if len(top_cg_pro_aicg_contact) > 0
            write(itp_file, itp_contact_head)
            write(itp_file, itp_contact_comm)
            for i_c in range(1, len(top_cg_pro_aicg_contact) + 1):
                printfmt(itp_file,
                         itp_contact_line,
                         top_cg_pro_aicg_contact[i_c][1],
                         top_cg_pro_aicg_contact[i_c][2],
                         AICG_CONTACT_FUNC_TYPE,
                         top_cg_pro_aicg_contact[i_c][3] * 0.1,
                         param_cg_pro_e_contact[i_c] * CAL2JOU)
            end
            write(itp_file, "\n")
        end

        # write RNA Go-type native contacts
        if len(top_cg_RNA_base_stack) + len(top_cg_RNA_base_pair) + len(top_cg_RNA_other_contact) > 0
            write(itp_file, itp_contact_head)
            write(itp_file, itp_contact_comm)
            for i_c in range(1, len(top_cg_RNA_base_stack) + 1):
                printfmt(itp_file,
                         itp_contact_line,
                         top_cg_RNA_base_stack[i_c][1],
                         top_cg_RNA_base_stack[i_c][2],
                         RNA_CONTACT_FUNC_TYPE,
                         top_cg_RNA_base_stack[i_c][3] * 0.1,
                         top_cg_RNA_base_stack[i_c][4] * CAL2JOU)
            end
            for i_c in range(1, len(top_cg_RNA_base_pair) + 1):
                printfmt(itp_file,
                         itp_contact_line,
                         top_cg_RNA_base_pair[i_c][1],
                         top_cg_RNA_base_pair[i_c][2],
                         RNA_CONTACT_FUNC_TYPE,
                         top_cg_RNA_base_pair[i_c][3] * 0.1,
                         top_cg_RNA_base_pair[i_c][4] * CAL2JOU)
            end
            for i_c in range(1, len(top_cg_RNA_other_contact) + 1):
                printfmt(itp_file,
                         itp_contact_line,
                         top_cg_RNA_other_contact[i_c][1],
                         top_cg_RNA_other_contact[i_c][2],
                         RNA_CONTACT_FUNC_TYPE,
                         top_cg_RNA_other_contact[i_c][3] * 0.1,
                         top_cg_RNA_other_contact[i_c][4] * CAL2JOU)
            end
            write(itp_file, "\n")
        end

        # write protein-RNA native contacts
        if len(top_cg_pro_RNA_contact) > 0
            write(itp_file, itp_contact_head)
            write(itp_file, itp_contact_comm)
            for i_c in range(1, len(top_cg_pro_RNA_contact) + 1):
                printfmt(itp_file,
                         itp_contact_line,
                         top_cg_pro_RNA_contact[i_c][1],
                         top_cg_pro_RNA_contact[i_c][2],
                         RNP_CONTACT_FUNC_TYPE,
                         top_cg_pro_RNA_contact[i_c][3] * 0.1,
                         top_cg_pro_RNA_contact[i_c][4] * CAL2JOU)
            end
            write(itp_file, "\n")
        end


        # ---------------------
        #        [ exclusions ]
        # ---------------------

        # write Protein exclusion list
        if len(top_cg_pro_aicg_contact) > 0
            write(itp_file, itp_exc_head)
            write(itp_file, itp_exc_comm)
            for i_c in range(1, len(top_cg_pro_aicg_contact) + 1):
                printfmt(itp_file,
                         itp_exc_line,
                         top_cg_pro_aicg_contact[i_c][1],
                         top_cg_pro_aicg_contact[i_c][2])
            end
            write(itp_file, "\n")
        end

        # write RNA exclusion list
        if len(top_cg_RNA_base_stack) + len(top_cg_RNA_base_pair) + len(top_cg_RNA_other_contact) > 0
            write(itp_file, itp_exc_head)
            write(itp_file, itp_exc_comm)
            for i_c in range(1, len(top_cg_RNA_base_stack) + 1):
                printfmt(itp_file,
                         itp_exc_line,
                         top_cg_RNA_base_stack[i_c][1],
                         top_cg_RNA_base_stack[i_c][2])
            end
            for i_c in range(1, len(top_cg_RNA_base_pair) + 1):
                printfmt(itp_file,
                         itp_exc_line,
                         top_cg_RNA_base_pair[i_c][1],
                         top_cg_RNA_base_pair[i_c][2])
            end
            for i_c in range(1, len(top_cg_RNA_other_contact) + 1):
                printfmt(itp_file,
                         itp_exc_line,
                         top_cg_RNA_other_contact[i_c][1],
                         top_cg_RNA_other_contact[i_c][2])
            end
            write(itp_file, "\n")
        end

        # write protein-RNA exclusion contacts
        if len(top_cg_pro_RNA_contact) > 0
            write(itp_file, itp_exc_head)
            write(itp_file, itp_exc_comm)
            for i_c in range(1, len(top_cg_pro_RNA_contact) + 1):
                printfmt(itp_file,
                         itp_exc_line,
                         top_cg_pro_RNA_contact[i_c][1],
                         top_cg_pro_RNA_contact[i_c][2])
            end
            write(itp_file, "\n")
        end



        close(itp_file)
        print(">           ... .itp: DONE!")
       
    end

    # ------------------------------------------------------------
    # gro ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ------------------------------------------------------------
    if do_output_gro
        # HEAD: time in the unit of ps
        GRO_HEAD_STR  = "{}, t= {:>16.3f} \n"
        # ATOM NUM: free format int
        GRO_ATOM_NUM  = "{:>12d} \n"
        # XYZ: in the unit of nm!!!
        GRO_ATOM_LINE = "{:>5d}{:>5}{:>5}{:>5d} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} \n"
        GRO_BOX_LINE  = "{:>15.4f}{:>15.4f}{:>15.4f} \n\n"

        gro_name = pdb_name[1:end-4] * "_cg.gro"
        gro_file = open(gro_name, "w")

        printfmt(gro_file, GRO_HEAD_STR, "CG model for GENESIS ", 0)
        printfmt(gro_file, GRO_ATOM_NUM, cg_num_particles)

        for i_bead in range(1, cg_num_particles + 1):
            printfmt(gro_file, GRO_ATOM_LINE,
                     cg_resid_index[i_bead],
                     cg_resid_name[i_bead],
                     cg_bead_name[i_bead],
                     i_bead,
                     cg_bead_coor[1 , i_bead] * 0.1,
                     cg_bead_coor[2 , i_bead] * 0.1,
                     cg_bead_coor[3 , i_bead] * 0.1,
                     0.0, 0.0, 0.0)
        end
        printfmt(gro_file, GRO_BOX_LINE, 0.0, 0.0, 0.0)

        close(gro_file)
        print(">           ... .gro: DONE!")
    end


    # ------------------------------------------------------------
    # PWMcos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ------------------------------------------------------------
    if do_output_pwmcos
        itp_pwmcos_head = "[ pwmcos ]\n"
        itp_pwmcos_comm     = format(";{:>5}{:>4}{:>9}{:>9}{:>9}{:>9}{:>12}{:>12}{:>12}{:>12}{:>8}{:>8}\n",
                                     "i", "f", "r0", "theta1", "theta2", "theta3",
                                     "ene_A", "ene_C", "ene_G", "ene_T",
                                     "gamma", "eps'")
        # itp_pwmcos_line = "{:>6d} {:>3d} {:>8.5f} {:>8.3f} {:>8.3f} {:>8.3f}{:>12.6f}{:>12.6f}{:>12.6f}{:>12.6f}{:>8.3f}{:>8.3f} \n"
        itp_pwmcos_line = "%6d %3d %8.5f %8.3f %8.3f %8.3f%12.6f%12.6f%12.6f%12.6f%8.3f%8.3f \n"

        if len( appendto_filename ) == 0
            itp_pwmcos_name = pdb_name[1:end-4] * "_cg_pwmcos.itp_patch"
            itp_pwmcos_file = open(itp_pwmcos_name, "w")
        else:
            itp_pwmcos_name = appendto_filename
            itp_pwmcos_file = open(itp_pwmcos_name, "a")
        end

        write(itp_pwmcos_file, itp_pwmcos_head)
        write(itp_pwmcos_file, itp_pwmcos_comm)
        for itpterm in top_cg_pro_DNA_pwmcos
            @printf(itp_pwmcos_file,
                    "%6d %3d %8.5f %8.3f %8.3f %8.3f%12.6f%12.6f%12.6f%12.6f%8.3f%8.3f \n",
                    itpterm[1],
                    PWMCOS_FUNC_TYPE,
                    itpterm[2] * 0.1,
                    itpterm[3],
                    itpterm[4],
                    itpterm[5],
                    itpterm[6],
                    itpterm[7],
                    itpterm[8],
                    itpterm[9],
                    pwmcos_gamma,
                    pwmcos_epsil)
        end
        write(itp_pwmcos_file, "\n")

        close(itp_pwmcos_file)
        print(">           ... ", itp_pwmcos_name, " pwmcos.itp: DONE!")
    end

    # ------------------------------------------------------------
    # psf ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ------------------------------------------------------------
    if do_output_psf
        psf_head_str = "PSF CMAP \n\n"
        psf_title_str0 = "      3 !NTITLE \n"
        psf_title_str1 = "REMARKS PSF file created with Julia. \n"
        psf_title_str2 = "REMARKS System: {1}  \n"
        psf_title_str5 = "REMARKS ======================================== \n"
        psf_title_str6 = "       \n"
        psf_title_str = psf_title_str0 * psf_title_str1 * psf_title_str2 * psf_title_str5 * psf_title_str6
        psf_atom_title = " {:>6d} !NATOM \n"
        # PSF_ATOM_LINE = " {atom_ser} {seg_id} {res_ser} {res_name} {atom_name} {atom_type}  {charge}  {mass}   0"
        psf_atom_line = " {:>6d} {:>3} {:>5d} {:>3} {:>3} {:>5}  {:>10.6f}  {:>10.6f}          0 \n"
        chain_id_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"

        psf_name = pdb_name[1:end-4] * "_cg.psf"
        psf_file = open(psf_name, "w")
        write(psf_file, psf_head_str)
        printfmt(psf_file, psf_title_str, pdb_name[1:end-4])
        printfmt(psf_file, psf_atom_title, cg_num_particles)
        for i_bead in range(1, cg_num_particles + 1):
            printfmt(psf_file,
                     psf_atom_line,
                     i_bead,
                     chain_id_set[cg_chain_id[i_bead]],
                     cg_resid_index[i_bead],
                     cg_resid_name[i_bead],
                     cg_bead_name[i_bead],
                     cg_bead_type[i_bead],
                     cg_bead_charge[i_bead],
                     cg_bead_mass[i_bead])
        end
        write(psf_file,"\n")

        close(psf_file)
        print(">           ... .psf: DONE!")
    end

    # ------------------------------------------------------------
    # cgpdb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ------------------------------------------------------------
    if do_output_cgpdb
        cg_pdb_name = pdb_name[1:end-4] * "_cg.pdb"
        cg_pdb_file = open(cg_pdb_name, "w")
        cg_pdb_atom_line = "ATOM  {:>5d} {:>4s}{:1}{:<4s}{:1}{:>4d}{:1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>10s}{:2s}{:2s} \n"
        chain_id_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
        tmp_chain_id = 0
        for i_bead in range(1, cg_num_particles + 1):
            if cg_chain_id[i_bead] > tmp_chain_id
                if tmp_chain_id > 0
                    write(cg_pdb_file, "TER\n")
                end
                tmp_chain_id = cg_chain_id[i_bead]
            end
            printfmt(cg_pdb_file,
                     cg_pdb_atom_line,
                     i_bead,
                     cg_bead_name[i_bead],
                     ' ',
                     cg_resid_name[i_bead],
                     chain_id_set[cg_chain_id[i_bead]],
                     cg_resid_index[i_bead],
                     ' ',
                     cg_bead_coor[1 , i_bead],
                     cg_bead_coor[2 , i_bead],
                     cg_bead_coor[3 , i_bead],
                     0.0,
                     0.0,
                     "",
                     "",
                     "")
        end
        write(cg_pdb_file,"TER\n")
        write(cg_pdb_file,"END\n")
        write(cg_pdb_file,"\n")

        close(cg_pdb_file)
        print(">           ... .pdb (CG) : DONE!")
    end

    # --------------------------------------------------------------
    # show sequence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------
    if do_output_sequence
        chain_id_set = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890"
        cg_seq_name = pdb_name[1:end-4] * "_cg.fasta"
        cg_seq_file = open(cg_seq_name, "w")

        for i_chain in range(aa_num_chain):
            chain = aa_chains[i_chain]
            mol_type = cg_chain_mol_types[i_chain]
            printfmt(cg_seq_file,
                     "> Chain {1} : {2} \n",
                     chain_id_set[i_chain],
                     MOL_TYPE_LIST[mol_type])

            for i_res in chain.residues
                res_name = aa_residues[i_res].name
                write(cg_seq_file, RES_SHORTNAME_DICT[res_name])
            end

            write(cg_seq_file, "\n")
        end

        close(cg_seq_file)
        print(">           ... sequence output : DONE!")
    end

    print("------------------------------------------------------------")
    print("------------------------------------------------------------")
    print("[1;32m FINISH! [0m ")
    print(" Please check the .itp and .gro files.")
    print("============================================================")
end

# =============================
# Parsing Commandline Arguments
# =============================
def parse_commandline():
    parser = argparse.ArgumentParser(description="Generate CG model top files for Genesis.")

    parser.add_argument('pdb', type=str, help="PDB file name")

    parser.add_argument('-c', '--respac', type=str, default="",
                        help="RESPAC protein charge distribution data.")
    parser.add_argument('--aicg-scale', type=int, default=1,
                        help="Scale AICG2+ local interactions: 0) average; 1) general (default).")
    parser.add_argument('--3spn-param', action='store_true',
                        help="Generate 3SPN.2C parameters from x3DNA generated PDB structure.")
    parser.add_argument('--pwmcos', action = 'store_True',
                         help = "Generate parameters for protein-DNA sequence-specific interactions.")
    parser.add_argument('--pwmcos-scale', type=float, default=1.0,
                        help = "Energy scaling factor for PWMcos.")
    parser.add_argument("--pwmcos-shift", type=float, default=0.0,
                        help = "Energy shifting factor for PWMcos.")
    parser.add_argument("--psf", action = 'store_True',
                        help = "Prepare PSF file.")
    parser.add_argument("--cgpdb", action = 'store_True',
                        help = "Prepare CG PDB file.")
    parser.add_argument("-p", "--pfm", type=str, default="",
                        help = "Position frequency matrix file for protein-DNA sequence-specific interactions.")
    parser.add_argument("--patch", type=str, default="",
                        help = "Append (apply patch) to .itp file.")
    parser.add_argument("--show-sequence", action = 'store_True',
                        help = "Show sequence of molecules in PDB.")
    parser.add_argument("--debug", action = 'store_True',
                        help = "DEBUG.")

    return parser.parse_args()

# ====
# Main
# ====

def main()

    args = parse_commandline()

    pdb_2_top(args)


if __name__ == '__main__':
    main()

