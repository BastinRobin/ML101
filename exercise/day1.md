## Data Exploration Using Parametric Methods

Parametric statistical methods often mean those methods that assume the data samples have a Gaussian distribution.

in applied machine learning, we need to compare data samples, specifically the mean of the samples. Perhaps to see if one technique performs better than another on one or more datasets. To quantify this question and interpret the results, we can use parametric hypothesis testing methods such as the Student’s t-test and ANOVA.

#### After completing these exercise, you will know:

- The Student’s t-test for quantifying the difference between the mean of two independent data samples.
- The paired Student’s t-test for quantifying the difference between the mean of two dependent data samples.
- The ANOVA and repeated measures ANOVA for checking the similarity or difference between the means of 2 or more data samples.

## Outline
- Parametric Statistical Significance Tests
- Test Data
- Student’s t-Test
- Paired Student t-Test
- Analysis of Variance Test
- Repeated Measures ANOVA Test

#### Parametric Statistical Significance Tests
They often refer to statistical tests that assume the Gaussian distribution. Because it is so common for data to fit this distribution, parametric statistical methods are more commonly used.

A typical question we may have about two or more samples of data is whether they have the same distribution. Parametric statistical significance tests are those statistical methods that assume data comes from the same Gaussian distribution, that is a data distribution with the same mean and standard deviation: the parameters of the distribution.

In general, each test calculates a test statistic that must be interpreted with some background in statistics and a deeper knowledge of the statistical test itself. Tests also return a p-value that can be used to interpret the result of the test. The p-value can be thought of as the probability of observing the two data samples given the base assumption (null hypothesis) that the two samples were drawn from a population with the same distribution.

The p-value can be interpreted in the context of a chosen significance level called alpha. A common value for alpha is 5%, or 0.05. If the p-value is below the significance level, then the test says there is enough evidence to reject the null hypothesis and that the samples were likely drawn from populations with differing distributions.

    
    p <= alpha: reject null hypothesis, different distribution.
    p > alpha: fail to reject null hypothesis, same distribution.


#### Test Data
Before we look at specific parametric significance tests, let’s first define a test dataset that we can use to demonstrate each test.

We will generate two samples drawn from different distributions. Each sample will be drawn from a Gaussian distribution.

We will use the randn() NumPy function to generate a sample of 100 Gaussian random numbers in each sample with a mean of 0 and a standard deviation of 1. Observations in the first sample are scaled to have a mean of 50 and a standard deviation of 5. Observations in the second sample are scaled to have a mean of 51 and a standard deviation of 5.

We expect the statistical tests to discover that the samples were drawn from differing distributions, although the small sample size of 100 observations per sample will add some noise to this decision.

    # generate gaussian data samples
    from numpy.random import seed
    from numpy.random import randn
    from numpy import mean
    from numpy import std
    # seed the random number generator
    seed(1)
    # generate two sets of univariate observations
    data1 = 5 * randn(100) + 50
    data2 = 5 * randn(100) + 51
    # summarize
    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))


Running the example generates the data samples, then calculates and prints the mean and standard deviation for each sample, confirming their different distribution.

    
    data1: mean=50.303 stdv=4.426
    data2: mean=51.764 stdv=4.660


#### Student’s t-Test

The Student’s t-test is a statistical hypothesis test that two independent data samples known to have a Gaussian distribution, have the same Gaussian distribution, named for William Gosset, who used the pseudonym “Student“.

One of the most commonly used t tests is the independent samples t test. You use this test when you want to compare the means of two independent samples on a given variable.

The assumption or null hypothesis of the test is that the means of two populations are equal. A rejection of this hypothesis indicates that there is sufficient evidence that the means of the populations are different, and in turn that the distributions are not equal.

- Fail to Reject H0: Sample distributions are equal.
- Reject H0: Sample distributions are not equal.

The Student’s t-test is available in Python via the ttest_ind() SciPy function. The function takes two data samples as arguments and returns the calculated statistic and p-value.

We can demonstrate the Student’s t-test on the test problem with the expectation that the test discovers the difference in distribution between the two independent samples. The complete code example is listed below.

    # Student's t-test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import ttest_ind
    # seed the random number generator
    seed(1)
    # generate two independent samples
    data1 = 5 * randn(100) + 50
    data2 = 5 * randn(100) + 51
    # compare samples
    stat, p = ttest_ind(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


Running the example calculates the Student’s t-test on the generated data samples and prints the statistic and p-value.

The interpretation of the statistic finds that the sample means are different, with a significance of at least 5%.

    Statistics=-2.262, p=0.025
    Different distributions (reject H0)


#### Paired Student’s t-Test

We may wish to compare the means between two data samples that are related in some way.

For example, the data samples may represent two independent measures or evaluations of the same object. These data samples are repeated or dependent and are referred to as paired samples or repeated measures.

Because the samples are not independent, we cannot use the Student’s t-test. Instead, we must use a modified version of the test that corrects for the fact that the data samples are dependent, called the paired Student’s t-test.

The test is simplified because it no longer assumes that there is variation between the observations, that observations were made in pairs, before and after a treatment on the same subject or subjects.

The default assumption, or null hypothesis of the test, is that there is no difference in the means between the samples. The rejection of the null hypothesis indicates that there is enough evidence that the sample means are different.

    Fail to Reject H0: Paired sample distributions are equal.
    Reject H0: Paired sample distributions are not equal.

The paired Student’s t-test can be implemented in Python using the ttest_rel() SciPy function. As with the unpaired version, the function takes two data samples as arguments and returns the calculated statistic and p-value.

We can demonstrate the paired Student’s t-test on the test dataset. Although the samples are independent, not paired, we can pretend for the sake of the demonstration that the observations are paired and calculate the statistic.


    # Paired Student's t-test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import ttest_rel
    # seed the random number generator
    seed(1)
    # generate two independent samples
    data1 = 5 * randn(100) + 50
    data2 = 5 * randn(100) + 51
    # compare samples
    stat, p = ttest_rel(data1, data2)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


Running the example calculates and prints the test statistic and p-value. The interpretation of the result suggests that the samples have different means and therefore different distributions.

    Statistics=-2.372, p=0.020
    Different distributions (reject H0)

#### Analysis of Variance Test

There are sometimes situations where we may have multiple independent data samples.

We can perform the Student’s t-test pairwise on each combination of the data samples to get an idea of which samples have different means. This can be onerous if we are only interested in whether all samples have the same distribution or not.

To answer this question, we can use the analysis of variance test, or ANOVA for short. ANOVA is a statistical test that assumes that the mean across 2 or more groups are equal. If the evidence suggests that this is not the case, the null hypothesis is rejected and at least one data sample has a different distribution.

    Fail to Reject H0: All sample distributions are equal.
    Reject H0: One or more sample distributions are not equal.

Importantly, the test can only comment on whether all samples are the same or not; it cannot quantify which samples differ or by how much.

The purpose of a one-way analysis of variance (one-way ANOVA) is to compare the means of two or more groups (the independent variable) on one dependent variable to see if the group means are significantly different from each other.

The test requires that the data samples are a Gaussian distribution, that the samples are independent, and that all data samples have the same standard deviation.

The ANOVA test can be performed in Python using the f_oneway() SciPy function. The function takes two or more data samples as arguments and returns the test statistic and f-value.

We can modify our test problem to have two samples with the same mean and a third sample with a slightly different mean. We would then expect the test to discover that at least one sample has a different distribution.

    # Analysis of Variance test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import f_oneway
    # seed the random number generator
    seed(1)
    # generate three independent samples
    data1 = 5 * randn(100) + 50
    data2 = 5 * randn(100) + 50
    data3 = 5 * randn(100) + 52
    # compare samples
    stat, p = f_oneway(data1, data2, data3)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


Running the example calculates and prints the test statistic and the p-value.

The interpretation of the p-value correctly rejects the null hypothesis indicating that one or more sample means differ.

    Statistics=3.655, p=0.027
    Different distributions (reject H0)


#### Repeated Measures ANOVA Test
We may have multiple data samples that are related or dependent in some way.

For example, we may repeat the same measurements on a subject at different time periods. In this case, the samples will no longer be independent; instead we will have multiple paired samples.

We could repeat the pairwise Student’s t-test multiple times. Alternately, we can use a single test to check if all of the samples have the same mean. A variation of the ANOVA test can be used, modified to test across more than 2 samples. This test is called the repeated measures ANOVA test.

The default assumption or null hypothesis is that all paired samples have the same mean, and therefore the same distribution. If the samples suggest that this is not the case, then the null hypothesis is rejected and one or more of the paired samples have a different mean.

    Fail to Reject H0: All paired sample distributions are equal.
    Reject H0: One or more paired sample distributions are not equal.
Repeated-measures ANOVA has a number of advantages over paired t tests, however. First, with repeated-measures ANOVA, we can examine differences on a dependent variable that has been measured at more than two time points, whereas with an independent t test we can only compare scores on a dependent variable from two time points.

