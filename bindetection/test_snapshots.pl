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

for $dir (sort @model_dirs)
{
	my $fulldir = $basedir."/".$dir;
	print "$fulldir ";
	if ((-f "$fulldir/mean.binaryproto") && (-f "$fulldir/labels.txt"))

	{
		#print " - database \n";
	}
	elsif (-f "$fulldir/deploy.prototxt")
	{
		print " - trained model\n";
		open (my $fh, "$fulldir/train_val.prototxt") || die "Could not open $fulldir/train_val.prototxt : $!";
		while ($line = <$fh>)
		{
			if ($line =~ /(\d{8}-\d{6}-[\da-f]{4})\/mean.binaryproto/)
			{
				print "Mean dir = $1\n";
				$mean_dir = $1;
				last;
			}
		}
		close ($fh);
		print "rm d12/*\n";
		print "cp $fulldir/* d12\n";
		print "cp $basedir/$mean_dir/mean.binaryproto d12\n";
		print "cp $basedir/$mean_dir/labels.txt d12\n";
		opendir(my $dh, "d12") || die "Can not open d12 : $!";
		my @snapshots = grep { /^snapshot_iter_/ && -f "d12/$_" } readdir($dh);
		closedir $dh;
		for $snapshot (sort @snapshots)
		{
			print "ln -sf $cwd/d12/$snapshot d12/network.caffemodel\n";
			for $video (sort @videos)
			{
				print "./zv --batch --groundTruth $videodir/$video\n";
			}
		}
	}
	else
	{
		print "\n";
	}

}

#for i in d12/snapshot_iter_*.caffemodel ; do ln -sf `pwd`/$i d12/network.caffemodel ;   ./zv --batch ~/ball_videos/dark\ purple/20160114_0.avi; echo $i; done

