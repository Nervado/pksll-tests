from scc_co2_buz.domain.models.well_data import IO

db_cols = [IO.PTPT,
           IO.TTPT,
           IO.TMONCKP,
           IO.PMONCKP,
           IO.ABERCKP,
           IO.ESTADO_POCO,
           IO.M1,
           IO.W1,
           IO.INIB
           ]

db_types = {
    'P-MON-CKP': float,
    'T-MON-CKP': float,
    'P-TPT': float,
    'T-TPT': float,
    'ABER-CKP':  float,
    'DIAG1': int,
    'PROB1': float,
    'DIAG2': int,
    'PROB2': float,
    'DIAG3': int,
    'PROB3': float,
    'GENERAL_DIAG': int,
    'HEART_BEAT': int,
    'ESTADO-POCO': int,
    'ESTADO-M1': int,
    'ESTADO-W1': int,
    'INIB': int}
