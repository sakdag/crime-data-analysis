MERGED_CRIME_DESC_CODES = [
    ('THEFT', 1),
    ('ASSAULT', 2),
    ('MINOR OFFENSE', 3),
    ('SEXUAL & CRIMES REGARDING CHILDREN', 4)
]

CRIME_LIST = [
    ['EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)', 'THEFT PLAIN - PETTY ($950 & UNDER)',
     'SHOPLIFTING - PETTY THEFT ($950 & UNDER)',
     'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD0036',
     'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $400 & UNDER',
     'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)', 'THEFT, PERSON', 'THEFT PLAIN - ATTEMPT',
     'THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)', 'THEFT OF IDENTITY', 'BUNCO, GRAND THEFT',
     'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)', 'BUNCO, PETTY THEFT',
     'THEFT FROM MOTOR VEHICLE - ATTEMPT',
     'BIKE - STOLEN', 'ROBBERY', 'BURGLARY', 'BURGLARY FROM VEHICLE', 'BURGLARY, ATTEMPTED',
     'BURGLARY FROM VEHICLE, ATTEMPTED', 'ATTEMPTED ROBBERY', 'VEHICLE - ATTEMPT STOLEN',
     'PURSE SNATCHING', 'CREDIT CARDS, FRAUD USE ($950.01 & OVER)'],
    ['INTIMATE PARTNER - SIMPLE ASSAULT', 'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
     'BATTERY - SIMPLE ASSAULT', 'INTIMATE PARTNER - AGGRAVATED ASSAULT',
     'BATTERY POLICE (SIMPLE)', 'DISCHARGE FIREARMS/SHOTS FIRED',
     'SHOTS FIRED AT INHABITED DWELLING', 'BRANDISH WEAPON', 'EXTORTION', 'CRIMINAL HOMICIDE'],
    ['STALKING', 'VANDALISM - MISDEAMEANOR ($399 OR UNDER)', 'TRESPASSING',
     'RESISTING ARREST', 'CONTEMPT OF COURT', 'DOCUMENT FORGERY / STOLEN FELONY',
     'VIOLATION OF TEMPORARY RESTRAINING ORDER', 'PROWLER', 'ARSON',
     'UNAUTHORIZED COMPUTER ACCESS', 'VIOLATION OF COURT ORDER',
     'VIOLATION OF RESTRAINING ORDER', 'FALSE IMPRISONMENT', 'CRUELTY TO ANIMALS',
     'DISTURBING THE PEACE', 'THROWING OBJECT AT MOVING VEHICLE',
     'CRIMINAL THREATS - NO WEAPON DISPLAYED',
     'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS) 0114',
     'THREATENING PHONE CALLS/LETTERS', 'PICKPOCKET'],
    ['SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH 0007=02',
     'SEXUAL PENTRATION WITH A FOREIGN OBJECT',
     'BATTERY WITH SEXUAL CONTACT',
     'SEX, UNLAWFUL', 'RAPE, FORCIBLE', 'ORAL COPULATION',
     'RAPE, ATTEMPTED',
     'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
     'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT', 'PEEPING TOM',
     'LEWD CONDUCT',
     'LETTERS, LEWD', 'INDECENT EXPOSURE',
     'CHILD ANNOYING (17YRS & UNDER)',
     'CHILD NEGLECT (SEE 300 W.I.C.)', 'CHILD STEALING', 'KIDNAPPING',
     'CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)0060']
]

VISUALIZATION_INPUTS = [
    ['pie', 'line', 'confusion', 'q'],
    ['DRNumber', 'TimeOccurred', 'CrimeCode', 'CrimeCodeDescription', 'VictimAge', 'VictimSex', 'VictimDescent',
     'PremiseCode', 'PremiseDescription', 'MonthOccurred', 'DayOfWeek']
]
