class Instantiator(object):
    def __init__(self, *args, **kwargs):
        self.templates_ = {}
        self.readTemplates(args)

    @property
    def basenames(self):
        return [tbasename for tbasename, tmpl in self.templates_.items()]

    def readTemplates(self, fnames):
        import os
        for templatefile in fnames:
            if not os.path.exists(templatefile):
                raise Exception("File {} does not exist".format(templatefile))
            tname = os.path.basename(templatefile)
            with open(templatefile, "r") as f:
                self.templates_[tname] = f.read()

    def instantiateTemplates(self, params, outdir):
        """
        Instantiate all templates with values taken from dict params
        and write to outdir.
        """
        import os

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            if not os.path.isdir(outdir):
                raise Exception("Desginated output directory {} is an existing file".format(outdir))

        for tbasename, tmpl in self.templates_.items():
            from string import Template
            TT = Template(tmpl)
            txt =  TT.substitute(**params)
            tname = os.path.join(outdir, tbasename)
            with open(tname, "w") as tf:
                tf.write(txt)

    def writeParams(self, params, fname):
        """
        Write parameter dictionary params to file.
        """
        with open(fname, "w") as pf:
            for k, v in list(params.items()):
                pf.write("{name} {val:e}\n".format(name=k, val=v))



if __name__ == "__main__":
    import os, sys
    if len(sys.argv[1:]) < 2:
        print("need more arguments")
        sys.exit(1)

    II = Instantiator(*[f for f in sys.argv[1:-1]])

    params = {"XXX":22, "YYY":20}

    outdir = sys.argv[-1]

    II.instantiateTemplates(params, outdir)
    II.writeParams(params, os.path.join(outdir, "params.dat"))
