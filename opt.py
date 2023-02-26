import optuna
from subprocess import *
from sys import argv

def objective(trial):
    sep = trial.suggest_int('sep',10,50)
    unknwon = trial.suggest_int('unknown',3000,10000)
    diff1 = trial.suggest_int('diff1',10,1000)
    ex_step = trial.suggest_int('ex_step',10,diff1)
    pred_border = trial.suggest_int('pred_border',20,1000)
    low_damage = trial.suggest_int('low_damage',10,1000)
    high_damage = trial.suggest_int('high_damage',low_damage,3000)
    score = 0
    for i in range(100):
        proc = run(f".\\tools\\tester.exe .\\target\\release\\ahc018.exe {sep} {unknwon} {diff1} {ex_step} {pred_border} {low_damage} {high_damage} < tools/in/{i:04d}.txt",shell=True,stdout=PIPE,stderr=PIPE,encoding='utf-8')
        cur_score = int(proc.stderr.split()[-1])
        score += cur_score
    return score

assert len(argv) == 2
t = int(argv[1])
study_name = 'ahc018_hyper_parameters'
study = optuna.create_study(study_name=study_name,
                            storage='sqlite:///optuna_study.db',
                            load_if_exists=True)
study.optimize(objective, timeout=t)
print(study.best_params)
