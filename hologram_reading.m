%% 2D TM mode simulation
% Always utilize SI units
% Reading
% Whitelight source simulation.
close all
gpuDeviceTable
GPU1 = gpuDevice(1);
%% Import figures
msource = imresize(imread('msource.png'),[950,950]);

%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter('20251026_flat_mirror_reading.mp4','MPEG-4' );
    writerObj.FrameRate = fps;
    open(writerObj);
end

%% Universal constants
epsilon0=8.854187817e-12;
mu0=12.566370e-7;
c=1/sqrt(epsilon0*mu0);

%% Plot definitions
brightness=0.1;  %brightness of plot
nt=8000; %total number os time steps
waitdisp=15; %wait to display

lengthx=5e-6;  %size of the image in x axis SI
lengthy=lengthx;  %size of the image in y axis SI

[sy,sx]=size(msource(:,:,1));

emkc = zeros(1000,3); % EM colormap
emkc(1,:) = [1 0 0];
emkc(1000,:) = [0 0 1];
r = linspace(1,0,499);
for j = 1:length(r)
    emkc(j,1) = r(j);
end
b = linspace(0,1,500);
for j = 1:length(b)
    emkc(500+j,3) = b(j);
end
% emkc=emkcolormap(100); % EM colormap
% emc=emcolormap(100); % EM colormap

%% Source constants
wavelength=linspace(305e-9,405e-9,10);
readout_wavelength = zeros(length(wavelength),1);
readout_intensity = zeros(length(wavelength),1);
for i = 1:length(wavelength)
    incident_wavelength = wavelength(i);
    omega=2*pi()*c./incident_wavelength;
    
    %% Cell size
    dx=lengthx/sx;  %cell size in x
    dy=lengthy/sy;  %cell size in y
    
    %% Coordinates definition
    xmin=-lengthx/2;
    ymin=-lengthy/2;
    xmax=xmin+lengthx;
    ymax=ymin+lengthy;
    
    x=(xmin+dx/2:dx:xmax-dx/2);
    y=flip((ymin+dx/2:dy:ymax-dy/2));
    
    [X,Y]=meshgrid(x,y);
    
    %% Source Mask
    source=double(msource(:,:,1))/255;
%     source=double(reading_source(:,:,1))/255;
    
    %% Permittivity
%     obj1=double(ball_bearing_diameter_50_angle_0(:,:,3))/255; % P1 layer
%     obj2=double(ball_bearing_diameter_50_angle_0(:,:,2))/255; % P2 layer
%     lens = double(focus_lens(:,:,1))/255;
    obj1 = hydrogel;
    obj1 = round(double(obj1(:,:,2))/255);
    obj2 = exposure_intensity./max(exposure_intensity,[],"all");
    %imagesc(x,y,obj2);

    RI0=1;  %Background refractive index
    alpha1=log(2)/100e-9;  %Absorbtion coeficient with half loss in 100 nm
    RI1=1+1i*c*alpha1/2/omega(1); %Refractive indexwith absorption coeficient
    RI2=1.37;    %Refractive index of P1
    RI3=1.64-RI2;  %Refractive index of P2
    RI_lens = 2;
    
    sabs=10;    %Thickness of the absorbing layer in pixels (sigma of Gaussian)
    alphab=log(100)/sabs/dx/0.3;    %Absorption after crossing the layer
    gsf=repmat(exp(-(-(sx-1)/2:(sx-1)/2).^2/sabs.^2),sy,1).*repmat(exp(-(-(sy-1)/2:(sy-1)/2).^2/sabs.^2).',1,sx);
    absmask=ones(size(X));
    absmask(2:end-1,2:end-1)=zeros(size(X)-2);
    absmask=ifftshift(abs(ifft2(fft2(absmask).*fft2(gsf))));
    absmask=absmask/max(absmask(:));
    
    RIbackground=1i*absmask*c*alphab/2/omega;    %Imaginary part
    % RIbackground=RIbackground+sqrt(real(RI0).^2-imag(RI0).^2+imag(RIbackground).^2);   %Real part to maintain speed
    
    RIall=RI0;
    % RIall=RIbackground;
    RIall=RIall.*(1-obj1)+RI2.*obj1;
    RIall=RIall.*(1-obj2)+RI3.*obj2;
    %RIall=RIall.*(1-lens)+RI_lens.*lens;
    
    RIbackground=1i*absmask*c*alphab/2/omega;    %Imaginary part
    RIall=RIbackground+sqrt(real(RIall).^2-imag(RIall).^2+imag(RIbackground).^2);   %Real part to maintain speed
    
    epsilonc=epsilon0*RIall.^2;
    epsilon=abs(real(epsilonc));    %Real permittivity
    sigma=omega*imag(epsilonc); %Imaginary permittivity (Conductivity)
    mu=mu0*ones(size(X));
    
    %% Time step
    cfl=1/sqrt(2); %Condition in Fourier domain
    dt=cfl*min(real(sqrt(epsilon(:).*mu(:))))*0.5*dx; %Courant criteria for minimum timestep
    
    %% Initialization
    Hx=zeros(size(X));
    Hy=zeros(size(X));
    Ez=zeros(size(X));
    
    JEz=zeros(size(Ez));
    
    ceb=1./(1+(sigma.*dt./(2*epsilon)));
    cea=(1-(sigma.*dt./(2*epsilon))).*ceb;
    
    chbx=1./(1+(sigma.*dt./(2*mu)));
    chax=(1-(sigma.*dt./(2*mu))).*chbx;  
    
    chby=1./(1+(sigma.*dt./(2*mu)));
    chay=(1-(sigma.*dt./(2*mu))).*chby;  
    
    %% Fourier constants for derivative
    xk = -1i*2*pi()*fftshift(-sx/2:sx/2-1)/sx;
    yk = flip(-1i*2*pi()*fftshift(-sy/2:sy/2-1)/sy);
    [Xk,Yk]=meshgrid(xk,yk);

    %% Convert to GPU object
    GHx = gpuArray(Hx);
    GHy = gpuArray(Hy);
    GEz = gpuArray(Ez);
    GJEz = gpuArray(JEz);
    Gchax = gpuArray(chax);
    Gchbx = gpuArray(chbx);
    Gchay = gpuArray(chay);
    Gchby = gpuArray(chby);
    Gcea = gpuArray(cea);
    Gceb = gpuArray(ceb);
    
    %% This is the field propagation loop
    ki=0;
    figure
    
    nt_end = 4000;
    % exposure_intensity = zeros(size(X));
    Ez_his = zeros(ki,1);
    GEz_his = gpuArray(Ez_his);
    peak = zeros(ki,2);
    Gpeak = gpuArray(peak);
    real_peak = [];
    Greal_peak = gpuArray(real_peak);
    reflection_wavelength = [];
    Greflection_wavelength = gpuArray(reflection_wavelength);
    tic
    while ki<nt
        ki=ki+1;
        %Define source
        GJEz=source*cos(omega*dt*ki);
        %JEz=source*cos(omega(ceil(ki/100))*dt*ki);
        
        %Update of Ez
        DyHx=real(ifft2(Yk.*fft2(GHx))/dy); % Fourier derivative
        DxHy=real(ifft2(Xk.*fft2(GHy))/dx); 
        
        GEz  =Gcea .* GEz   + Gceb .* (dt./epsilon) .* (DyHx-DxHy) + GJEz;
        
        %Update of Hx,Hy    
        DyEz=real(ifft2(Yk.*fft2(GEz))/dy); 
        DxEz=real(ifft2(Xk.*fft2(GEz))/dx);
        
        GHx = Gchax .* GHx + Gchbx .* (dt./mu).*(DyEz);    
        GHy = Gchay .* GHy + Gchby .* (dt./mu).*(-DxEz);
        
        GEz_his(ki,1) = GEz(100,400);
        if (ki >= 2) && (GEz(100,400)^2 <= GEz_his(ki-1,1)^2) && (GEz_his(ki-1,1)^2 >= GEz_his(ki-2,1)^2)
            Gpeak(ki-1,1) = dt*(ki-1);
            Gpeak(ki-1,2) = GEz_his(ki-1,1);
        end
        Greal_peak = Gpeak(Gpeak(:,2)~=0,:);
        if length(Greal_peak(:,1)) >= 2
            Greflection_wavelength(end+1) = c*(Greal_peak(end,1) - Greal_peak(end-1,1))*2;
        end
        if ~mod(ki,waitdisp)
            wait(GPU1);
            disp(['Time per cycle: ' num2str(toc/waitdisp) ' s']);
            clf
            Ez = gather(GEz);
            Hx = gather(GHx);
            Hy = gather(GHy);
            Ez_his = gather(GEz_his);
            peak = gather(Gpeak);
            real_peak = gather(Greal_peak);
            reflection_wavelength = gather(Greflection_wavelength);
            subplot(2,2,1);
            imagesc(x,y,Ez,[-1 1]/brightness)  
            hold on
            plot(x(100),y(400),'ro')
            axis image
            title(sprintf('TimeFrame %s',num2str(ki)))
            set(gca,'ydir','normal' )
            colormap(emkc)
            alpha(1-obj1*0.1-obj2*0.15)
    
            subplot(2,2,2);
            plot(dt:dt:ki*dt,Ez_his(1:ki))
            xlabel('time (second)')
            ylabel('Ez Field')
            
            subplot(2,2,3);
            plot(dt:dt:ki*dt,Ez_his(1:ki).^2)
            hold on
            plot(peak(peak(:,2)~=0,1),peak(peak(:,2)~=0,2).^2)
            xlabel('time (second)')
            ylabel('Intensity')
            hold off
    
            subplot(2,2,4);
            plot(reflection_wavelength)
            xlabel('time (second)')
            ylabel('wavelength (m)')
    
            drawnow
            if movvar==1
                frame = getframe(gcf);
                writeVideo(writerObj,frame);
            end
            tic
        end
    end
    readout_wavelength(i) = reflection_wavelength(end);
    readout_intensity_temp = peak(peak(:,2)~=0,2).^2;
    readout_intensity(i) = readout_intensity_temp(end);
end
%% Close video
if movvar==1
    close(writerObj);
end
figure
plot(readout_wavelength,readout_intensity,'r-')
xlabel('Wavelength (nm)')
ylabel('Intensity')
