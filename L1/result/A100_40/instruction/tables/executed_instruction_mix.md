# L1 v2 — NCU Executed Instruction Mix

Counts are grouped from the NCU **Source** page by SASS opcode family. `Predicated-on thread instructions` is the most useful column for actual predicate-true work.

## load-xor r1

| SASS opcode | Warp instructions executed | Thread instructions executed | Predicated-on thread instructions |
| --- | ---: | ---: | ---: |
| LDG | 6750203904 | 216006524928 | 216006524928 |
| LOP3 | 3375101952 | 108003262464 | 108003262464 |
| ISETP | 949250880 | 30376028160 | 30376028160 |
| IADD3 | 949247424 | 30375917568 | 30375917568 |
| BRA | 949249152 | 30375972864 | 23625658368 |
| IMAD | 105484032 | 3375489024 | 3375489024 |
| USHF | 5184 | 165888 | 165888 |
| ULDC | 3456 | 110592 | 110592 |
| EXIT | 1728 | 55296 | 55296 |
| S2R | 1728 | 55296 | 55296 |

## load-xor r2

| SASS opcode | Warp instructions executed | Thread instructions executed | Predicated-on thread instructions |
| --- | ---: | ---: | ---: |
| LDG | 13500407808 | 432013049856 | 432013049856 |
| LOP3 | 6750203904 | 216006524928 | 216006524928 |
| ISETP | 949250880 | 30376028160 | 30376028160 |
| IADD3 | 949247424 | 30375917568 | 30375917568 |
| BRA | 949249152 | 30375972864 | 20250556416 |
| IMAD | 105484032 | 3375489024 | 3375489024 |
| USHF | 5184 | 165888 | 165888 |
| ULDC | 3456 | 110592 | 110592 |
| EXIT | 1728 | 55296 | 55296 |
| S2R | 1728 | 55296 | 55296 |

## xor-only r1

| SASS opcode | Warp instructions executed | Thread instructions executed | Predicated-on thread instructions |
| --- | ---: | ---: | ---: |
| LOP3 | 3375101952 | 108003262464 | 108003262464 |
| ISETP | 949250880 | 30376028160 | 30376028160 |
| BRA | 949249152 | 30375972864 | 23625658368 |
| IADD3 | 527359680 | 16875509760 | 16875509760 |
| IMAD | 105478848 | 3375323136 | 3375323136 |
| EXIT | 1728 | 55296 | 55296 |
| S2R | 1728 | 55296 | 55296 |

## xor-only r2

| SASS opcode | Warp instructions executed | Thread instructions executed | Predicated-on thread instructions |
| --- | ---: | ---: | ---: |
| LOP3 | 6750203904 | 216006524928 | 216006524928 |
| ISETP | 949250880 | 30376028160 | 30376028160 |
| BRA | 949249152 | 30375972864 | 20250556416 |
| IADD3 | 527359680 | 16875509760 | 16875509760 |
| IMAD | 105478848 | 3375323136 | 3375323136 |
| EXIT | 1728 | 55296 | 55296 |
| S2R | 1728 | 55296 | 55296 |

