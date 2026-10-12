import sys, os
R=sys.argv[1]; which=sys.argv[2]
A=os.path.join(R,'audit')
if which=='collect':
    src=open(os.path.join(R,'paper','collect.py'),encoding='utf-8').read()
    src=src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = r'{A}'").replace("ROOT = os.path.dirname(HERE)", f"ROOT = r'{R}'")
    src=src.replace("'results_tables.md'","'c_tables.md'").replace("'results_all.json'","'c_all.json'")
else:
    src=open(os.path.join(R,'exp2','analyze2.py'),encoding='utf-8').read()
    src=src.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = r'{os.path.join(R,'exp2')}'")
    src=src.replace("open(os.path.join(HERE, 'RESULTS2_tables.md'), 'w', encoding='utf-8')", f"open(r'{A}/a2_tables.md','w',encoding='utf-8')").replace("open(os.path.join(HERE, 'results2.json'), 'w')", f"open(r'{A}/a2.json','w')")
    assert "RESULTS2_tables.md'), 'w'" not in src and "results2.json'), 'w'" not in src
src=src.replace("if __name__ == '__main__':","if True:")
exec(compile(src,'x','exec'),{'__name__':'x'})
