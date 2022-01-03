if __name__ == "__main__":
    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-o", dest="OUTPUT", default="data.csv", help="Output data (default: %default)")
    op.add_option("-n", "--npoints", dest="NRUNS", default=6, type=int, help="Number of measurements (default: %default)")
    op.add_option("-c", "--compiler", dest="COMPILER", default="dpcpp", help="The compiler command (default: %default)")
    op.add_option("-e", "--exec", dest="EXEC", default="./a.out 2000", help="The run command (default: %default)")
    op.add_option("-f", "--flags", dest="FLAGS", default="-Ofast -DNOGPU", help="The C flags (default: %default)")
    opts, args = op.parse_args()

    import explorer

    II = explorer.Instantiator(*args)

    results = []
    outdir = "testout"

    for i in range (1,10):
        params = {"XXX":20, "YYY":20, "GX":2, "GY":i}
        II.instantiateTemplates(params, outdir)
        ee = explorer.Experiment(outdir, *[os.path.join(outdir, f) for f in II.basenames],
                compiler=opts.COMPILER,
                execstring=opts.EXEC,
                cflags=opts.FLAGS.split()
                )
        ee.compile()
        try:
            data, mu, sigma = ee.run(opts.NRUNS)
            results.append((params, data, mu,sigma))
        except:
            print("no data for {}".format(i))
            pass


    pnames = sorted([i for i in results[0][0].keys()])

    # Write tidy data csv
    header=",".join(pnames) +",t,mu,sigma\n"
    with open(opts.OUTPUT, "w") as f:
        f.write(header)
        for r in results:
            temp  = ",".join(map(str, [r[0][p] for p in pnames]))
            for sr in r[1]:
                temp2 = str(sr)
                f.write("{},{},{},{}\n".format(temp, temp2, r[2], r[3]))

