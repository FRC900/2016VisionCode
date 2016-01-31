#!/usr/bin/perl
#
use Cwd 'abs_path';
my $cwd = abs_path(".");

my $videodir = "/home/kjaget/ball_videos/test";
my $basedir = "/home/kjaget/jobs";

opendir(my $dh, $basedir) || die "Can not open $basedir : $!";
my @model_dirs = grep { /^201/ && -d "$basedir/$_" } readdir($dh);
closedir $dh;

opendir(my $dh, $videodir) || die "Can not open $videodir : $!";
my @videos = grep { /.avi$/ && -f "$videodir/$_" } readdir($dh);
closedir $dh;

for $video (sort @videos)
{
	print "$video\n";
}

for $dir (sort @model_dirs)
{
	my $fulldir = $basedir."/".$dir;
	#print "$fulldir ";
	if ((-f "$fulldir/mean.binaryproto") && (-f "$fulldir/labels.txt"))

	{
	}
	elsif (-f "$fulldir/deploy.prototxt")
	{
		open (my $fh, "$fulldir/train_val.prototxt") || die "Could not open $fulldir/train_val.prototxt : $!";
		while (my $line = <$fh>)
		{
			if ($line =~ /(\d{8}-\d{6}-[\da-f]{4})\/mean.binaryproto/)
			{
				#print "Mean dir = $1\n";
				$mean_dir = $1;
			}
			elsif ($line =~ /type: "Convolution"/)
			{
				while (my $line = <$fh>)
				{
					if ($line =~ /num_output: (\d+)/)
					{
						$conv_out = $1;
						last;
					}
				}
			}
			elsif ($line =~ /type: "InnerProduct"/)
			{
				while (my $line = <$fh>)
				{
					if ($line =~ /num_output: (\d+)/)
					{
						$fc_out = $1;
						last;
					}
				}
				last;
			}
		}
		close ($fh);
		open (my $fh, "$fulldir/solver.prototxt") || die "Could not open $fulldir/solver.prototxt : $!";
		while (my $line = <$fh>)
		{
			if ($line =~ /base_lr: ([\d\.e-]+)/)
			{
				$base_lr = $1;
			}
		}
		close ($fh);
		`rm d12/*`;
		`cp $fulldir/* d12`;
		`cp $basedir/$mean_dir/mean.binaryproto d12`;
		`cp $basedir/$mean_dir/labels.txt d12`;
		opendir(my $dh, "d12") || die "Can not open d12 : $!";
		my @snapshots = grep { /^snapshot_iter_/ && -f "d12/$_" } readdir($dh);
		closedir $dh;
		for $snapshot (sort @snapshots)
		{
			`ln -sf $cwd/d12/$snapshot d12/network.caffemodel`;
			print "$conv_out, $fc_out, $base_lr, $fulldir, $snapshot, ";
			print "\n";
			last;
			for $video (sort @videos)
			{
				open (my $pipeh, "./zv --batch --groundTruth $videodir/$video |");
				while ($line = <$pipeh>)
				{
					if ($line =~ /(\d+) of (\d+) ground truth objects/)
					{
						print "$1, $2, ";
					}
					elsif ($line =~ /(\d+) false positives found in (\d+) frames/)
					{
						print "$1, $2, ";
				}
				close $pipeh;
			}
			print "\n";
		}
	}
	else
	{
		print "\n";
	}

}

