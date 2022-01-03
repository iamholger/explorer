def kwery(kwargs, key, default):
    return kwargs[key] if kwargs.get(key) is not None else default


class Experiment(object):
    def __init__(self, *args, **kwargs):
        self.sourcefiles_ = args[1:]
        self.compiler_     = kwery(kwargs, "compiler"  ,  "dpcpp")
        self.executable_   = kwery(kwargs, "executable",  "a.out")
        self.execstring_   = kwery(kwargs, "execstring", "./a.out 2000")
        self.cflags_       = kwery(kwargs, "cflags"    , ["-Ofast", "-DNOGPU"])
        self.ldflags_      = kwery(kwargs, "ldflags"   , [])

    def compile(self):
        import subprocess
        cmd = self.compiler_  + " " + " ".join(self.cflags_) + " ".join(self.ldflags_) + " " +   " ".join(self.sourcefiles_)  + " -o " + self.executable_
        proc = subprocess.run(cmd.split())

    def measure(self):
        import subprocess
        cmd = self.execstring_
        proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        t=proc.stdout.splitlines()[-1].split()[0]
        return int(t)

    def run(self, niterations):
        data = [self.measure() for i in range(niterations)]
        import statistics
        mu, sigma = statistics.mean(data), statistics.stdev(data)
        print("Measurement: %.3f +/- %.3f"%(mu, sigma))
        return data, mu, sigma


if __name__ == "__main__":
    import sys
    ee = Experiment(sys.argv[1], sys.argv[2], compiler=sys.argv[4])
    ee.compile()
    data, mu, sigma = ee.run(int(sys.argv[3]))
