package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/// Fidelity models fitted to bootstrap resamples of the measured candidates, each shrinking towards its own
/// random point of the prior (see [FidelityModel#fit(List, double, Priors, int, long, double)]): their mean is
/// the prediction, their spread how unsure it is. Where the data decides a weight the members agree; where it
/// does not they differ.
public final class Ensemble {

    /// The prior perturbation of the members.
    public static final double PERTURB = 0.3;

    private final List<FidelityModel> members;

    private Ensemble(List<FidelityModel> members) {
        this.members = members;
    }

    /// `members` models fitted to resamples of `data` drawn from `seed`.
    public static Ensemble fit(List<Measured> data, int members, long seed, double ridge, Priors priors) {
        return fit(data, members, seed, ridge, priors, FidelityModel.DEFAULT_BANDS);
    }

    public static Ensemble fit(List<Measured> data, int members, long seed, double ridge, Priors priors, int bands) {
        List<Supplier<FidelityModel>> fits = fits(data, members, seed, ridge, priors, bands);
        List<FidelityModel> fitted = new ArrayList<>();
        for (Supplier<FidelityModel> fit : fits) fitted.add(fit.get());
        return new Ensemble(List.copyOf(fitted));
    }

    /// Where an ensemble's fits are run as small tasks: each call returns the future of one task.
    public interface Frames {
        <T> CompletableFuture<T> submit(Supplier<T> task);
    }

    /// As [#fit(List, int, long, double, Priors, int)], each member's fit a task of its own on `frames` and the
    /// ensemble joined in one more. The resamples and the members' seeds are drawn in order before any task
    /// runs, so the members are the sequential fit's, whichever order the tasks finish in.
    public static CompletableFuture<Ensemble> fit(
            List<Measured> data, int members, long seed, double ridge, Priors priors, int bands, Frames frames) {
        List<Supplier<FidelityModel>> fits = fits(data, members, seed, ridge, priors, bands);
        List<CompletableFuture<FidelityModel>> fitted = new ArrayList<>();
        for (Supplier<FidelityModel> fit : fits) fitted.add(frames.submit(fit));
        return CompletableFuture.allOf(fitted.toArray(CompletableFuture[]::new))
                .thenCompose(done -> frames.submit(() -> {
                    List<FidelityModel> models = new ArrayList<>();
                    for (var future : fitted) models.add(future.join());
                    return new Ensemble(List.copyOf(models));
                }));
    }

    /// The fit of each member to its bootstrap resample, the resamples and seeds drawn from `seed` in member order.
    private static List<Supplier<FidelityModel>> fits(
            List<Measured> data, int members, long seed, double ridge, Priors priors, int bands) {
        if (members < 1) throw new IllegalArgumentException("an ensemble needs a member");
        Random random = new Random(seed);
        List<Supplier<FidelityModel>> fits = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            List<Measured> resample = new ArrayList<>(data.size());
            for (int i = 0; i < data.size(); i++) resample.add(data.get(random.nextInt(data.size())));
            long memberSeed = random.nextLong();
            fits.add(() -> FidelityModel.fit(resample, ridge, priors, bands, memberSeed, PERTURB));
        }
        return fits;
    }

    public List<FidelityModel> members() {
        return members;
    }

    /// A member, for the objects and options all members share.
    public FidelityModel layout() {
        return members.get(0);
    }

    /// The mean of the members' predictions.
    public double mean(Allocation allocation) {
        double sum = 0;
        for (FidelityModel member : members) sum += member.predict(allocation);
        return sum / members.size();
    }

    /// The standard deviation of the members' predictions.
    public double spread(Allocation allocation) {
        double mean = mean(allocation), squares = 0;
        for (FidelityModel member : members) {
            double difference = member.predict(allocation) - mean;
            squares += difference * difference;
        }
        return Math.sqrt(squares / members.size());
    }

    /// The mean of the members' predictions of the soft assignment `p`.
    public double meanSoft(double[][] p) {
        double sum = 0;
        for (FidelityModel member : members) sum += member.predictSoft(p);
        return sum / members.size();
    }

    /// The derivative of [#meanSoft].
    public double[][] gradient(double[][] p) {
        double[][] sum = null;
        for (FidelityModel member : members) {
            double[][] gradient = member.gradient(p);
            if (sum == null) sum = gradient;
            else
                for (int i = 0; i < sum.length; i++)
                    for (int k = 0; k < sum[i].length; k++) sum[i][k] += gradient[i][k];
        }
        for (double[] row : sum) for (int k = 0; k < row.length; k++) row[k] /= members.size();
        return sum;
    }

    /// The mean over `unseen` of the absolute difference of the log of the mean prediction and of the measured
    /// KL; 0 for none.
    public double heldOutError(List<Measured> unseen) {
        if (unseen.isEmpty()) return 0;
        double sum = 0;
        for (Measured m : unseen)
            sum += Math.abs(Math.log(Math.max(mean(m.a()), 1e-9)) - Math.log(Math.max(m.kl(), 1e-9)));
        return sum / unseen.size();
    }
}
