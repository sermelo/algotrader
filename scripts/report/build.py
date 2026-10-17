import sys
sp=sys.argv[1]
t=open(f"{sp}/report_template.html").read()
d=open(f"{sp}/report_data.json").read().replace("</","<\\/")
open(f"{sp}/strategy-scorecard.html","w").write(t.replace("__DATA__",d))
