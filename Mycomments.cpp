//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::distortPoints

void cv::fisheye::distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha)
{
    CV_INSTRUMENT_REGION();

    // will support only 2-channel data now for points
    CV_Assert(undistorted.type() == CV_32FC2 || undistorted.type() == CV_64FC2);
    distorted.create(undistorted.size(), undistorted.type());
    //jgg: the number of InputArray
    size_t n = undistorted.total();

    //jgg: it only works when all conditions are met, otherwise there's a assert.
    CV_Assert(K.size() == Size(3,3) && (K.type() == CV_32F || K.type() == CV_64F) && D.total() == 4);

    //jgg: the focal length fx & fy, and cx & cy
    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0 ,2), camMat(1, 2));
    }

    //jgg: k is the distortion coefficient
    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    //jgg: the src point & the dst point
    const Vec2f* Xf = undistorted.getMat().ptr<Vec2f>();
    const Vec2d* Xd = undistorted.getMat().ptr<Vec2d>();
    Vec2f *xpf = distorted.getMat().ptr<Vec2f>();
    Vec2d *xpd = distorted.getMat().ptr<Vec2d>();

    for(size_t i = 0; i < n; ++i)
    {
        //jgg： get the src point
        Vec2d x = undistorted.depth() == CV_32F ? (Vec2d)Xf[i] : Xd[i];

        //jgg: x与x的点乘，开完根号即x的模
        double r2 = x.dot(x);
        double r = std::sqrt(r2);

        /*jgg: 镜头畸变相关，暂时不管，后面的thetam都是theta的m次方，这一块的公式具体可参考
        https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html中的Detailed Description部分
        */
        // Angle of the incoming ray:
        double theta = atan(r);

        double theta2 = theta*theta, theta3 = theta2*theta, theta4 = theta2*theta2, theta5 = theta4*theta,
                theta6 = theta3*theta3, theta7 = theta6*theta, theta8 = theta4*theta4, theta9 = theta8*theta;

        double theta_d = theta + k[0]*theta3 + k[1]*theta5 + k[2]*theta7 + k[3]*theta9;

        //jgg: calculate the  theta_d/r
        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        //jgg: calculate the  x'
        Vec2d xd1 = x * cdist;
        //jgg: calculate the  x'+alpha*y'
        Vec2d xd3(xd1[0] + alpha*xd1[1], xd1[1]);
        //jgg: calculate the  (u,v)
        Vec2d final_point(xd3[0] * f[0] + c[0], xd3[1] * f[1] + c[1]);

        if (undistorted.depth() == CV_32F)
            xpf[i] = final_point;
        else
            xpd[i] = final_point;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints

void cv::fisheye::undistortPoints( InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray R, InputArray P)
{
    CV_INSTRUMENT_REGION();

    // will support only 2-channel data now for points
    CV_Assert(distorted.type() == CV_32FC2 || distorted.type() == CV_64FC2);
    undistorted.create(distorted.size(), distorted.type());

    CV_Assert(P.empty() || P.size() == Size(3, 3) || P.size() == Size(4, 3));
    CV_Assert(R.empty() || R.size() == Size(3, 3) || R.total() * R.channels() == 3);
    CV_Assert(D.total() == 4 && K.size() == Size(3, 3) && (K.depth() == CV_32F || K.depth() == CV_64F));

    cv::Vec2d f, c;
    if (K.depth() == CV_32F)
    {
        Matx33f camMat = K.getMat();
        f = Vec2f(camMat(0, 0), camMat(1, 1));
        c = Vec2f(camMat(0, 2), camMat(1, 2));
    }
    else
    {
        Matx33d camMat = K.getMat();
        f = Vec2d(camMat(0, 0), camMat(1, 1));
        c = Vec2d(camMat(0, 2), camMat(1, 2));
    }

    Vec4d k = D.depth() == CV_32F ? (Vec4d)*D.getMat().ptr<Vec4f>(): *D.getMat().ptr<Vec4d>();

    cv::Matx33d RR = cv::Matx33d::eye();
    if (!R.empty() && R.total() * R.channels() == 3)
    {
        cv::Vec3d rvec;
        R.getMat().convertTo(rvec, CV_64F);
        RR = cv::Affine3d(rvec).rotation();
    }
    else if (!R.empty() && R.size() == Size(3, 3))
        R.getMat().convertTo(RR, CV_64F);

    if(!P.empty())
    {
        cv::Matx33d PP;
        P.getMat().colRange(0, 3).convertTo(PP, CV_64F);
        RR = PP * RR;
    }

    // start undistorting
    const cv::Vec2f* srcf = distorted.getMat().ptr<cv::Vec2f>();
    const cv::Vec2d* srcd = distorted.getMat().ptr<cv::Vec2d>();
    cv::Vec2f* dstf = undistorted.getMat().ptr<cv::Vec2f>();
    cv::Vec2d* dstd = undistorted.getMat().ptr<cv::Vec2d>();

    size_t n = distorted.total();
    int sdepth = distorted.depth();

    for(size_t i = 0; i < n; i++ )
    {
        Vec2d pi = sdepth == CV_32F ? (Vec2d)srcf[i] : srcd[i];  // image point
        Vec2d pw((pi[0] - c[0])/f[0], (pi[1] - c[1])/f[1]);      // world point

        double scale = 1.0;

        double theta_d = sqrt(pw[0]*pw[0] + pw[1]*pw[1]);

        // the current camera model is only valid up to 180 FOV
        // for larger FOV the loop below does not converge
        // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = min(max(-CV_PI/2., theta_d), CV_PI/2.);

        if (theta_d > 1e-8)
        {
            // compensate distortion iteratively
            double theta = theta_d;

            const double EPS = 1e-8; // or std::numeric_limits<double>::epsilon();
            for (int j = 0; j < 10; j++)
            {
                double theta2 = theta*theta, theta4 = theta2*theta2, theta6 = theta4*theta2, theta8 = theta6*theta2;
                double k0_theta2 = k[0] * theta2, k1_theta4 = k[1] * theta4, k2_theta6 = k[2] * theta6, k3_theta8 = k[3] * theta8;
                /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
                double theta_fix = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) /
                                   (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8);
                theta = theta - theta_fix;
                if (fabs(theta_fix) < EPS)
                    break;
            }

            scale = std::tan(theta) / theta_d;
        }

        Vec2d pu = pw * scale; //undistorted point

        // reproject
        Vec3d pr = RR * Vec3d(pu[0], pu[1], 1.0); // rotated point optionally multiplied by new camera matrix
        Vec2d fi(pr[0]/pr[2], pr[1]/pr[2]);       // final

        if( sdepth == CV_32F )
            dstf[i] = fi;
        else
            dstd[i] = fi;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::undistortPoints
