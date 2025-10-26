%% TM mode simulation
% Always utilize SI units
% Reading
% Whitelight source simulation.
close all
gpuDeviceTable
GPU1 = gpuDevice(1);
%% Import figures
msource = imresize(imread('reading_source.png'),[570,570]);
shelter = imresize(imread('source_part.png'),[570,570]);
lens = imresize(imread('lens.png'),[570,570]);
%% Initialize video
movvar=0;
fps=30;
if movvar==1
    writerObj = VideoWriter('20251026_flat_mirror_reading_farfieldFT.mp4','MPEG-4' );
    writerObj.FrameRate = fps;
    open(writerObj);
end

%% Universal constants
epsilon0=8.854187817e-12;
mu0=12.566370e-7;
c=1/sqrt(epsilon0*mu0);

%% Plot definitions
brightness=0.1;  %brightness of plot
nt=10000; %total number os time steps
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
wavelength=[290e-9, 355e-9, 500e-9];%linspace(295e-9,565e-9,10);
readout_wavelength_initial = zeros(length(wavelength),1);
readout_wavelength_final = zeros(length(wavelength),1);
readout_intensity = zeros(length(wavelength),1);
readout_transmission_initial = zeros(length(wavelength),1);
readout_transmission_final = zeros(length(wavelength),1);
theta_bin_edges = linspace(0,90,181);
theta_bin_centers = (theta_bin_edges(1:end-1)+theta_bin_edges(2:end))/2;
farfield_angular_intensity = NaN(length(wavelength),numel(theta_bin_centers));
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
    Gsource = gpuArray(source);
%     source=double(reading_source(:,:,1))/255;
    
    %% Permittivity
%     obj1=double(ball_bearing_diameter_50_angle_0(:,:,3))/255; % P1 layer
%     obj2=double(ball_bearing_diameter_50_angle_0(:,:,2))/255; % P2 layer
%     lens = double(focus_lens(:,:,1))/255;
    obj1 = hydrogel;
    obj1 = round(double(obj1(:,:,2))/255);
    % obj1 = zeros(570,570);
    % obj2 = zeros(570,570); %
    obj2 = exposure_intensity./max(exposure_intensity,[],"all");
    obj3 = shelter;
    obj3 = round(double(obj3(:,:,3))/255);
    obj4 = lens;
    obj4 = round(double(obj4(:,:,1))/255);
    %imagesc(x,y,obj2);

    RI0=1;  %Background refractive index
    alpha1=log(2)/100e-9;  %Absorbtion coeficient with half loss in 100 nm
    RI1=1+1i*c*alpha1/2/omega(1); %Refractive indexwith absorption coeficient
    RI2=1.37;    %Refractive index of P1
    RI3=1.64-RI2;  %Refractive index of P2
    RI4=10+1i*5;    
    RI_lens = 2;
    
    sabs=10;    %Thickness of the absorbing layer in pixels (sigma of Gaussian)
    alphab=log(100)/sabs/dx/0.3;    %Absorption after crossing the layer
    gsf=repmat(exp(-(-(sx-1)/2:(sx-1)/2).^2/sabs.^2),sy,1).*repmat(exp(-(-(sy-1)/2:(sy-1)/2).^2/sabs.^2).',1,sx);
    absmask=ones(size(X));
    absmask(2:end-1,2:end-1)=zeros(size(X)-2);
    absmask=ifftshift(abs(ifft2(fft2(absmask).*fft2(gsf))));
    absmask=absmask/max(absmask(:))+obj3;
    
    % RIbackground=1i*absmask*c*alphab/2/omega;    %Imaginary part
    % RIbackground=RIbackground+sqrt(real(RI0).^2-imag(RI0).^2+imag(RIbackground).^2);   %Real part to maintain speed
    
    RIall=RI0;
    % RIall=RIbackground;
    RIall=RIall.*(1-obj1)+RI2.*obj1;
    RIall=RIall.*(1-obj2)+RI3.*obj2;
    %RIall=RIall.*(1-obj3)+RI4.*obj3;
    RIall=RIall.*(1-obj4)+RI_lens.*obj4;
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

    %% Detector definition (20x20 pixel average)
    detector_size = 20;
    detector_half_size = floor(detector_size/2);
    detector_center_xy = [60,210]; % [column (x-index), row (y-index)] as referenced in plots
    % Keeping the x-index first matches the visual selection that was used
    % previously (plotting with plot(x(100), y(400))). We convert this to
    % [row, column] indices below so that the sampled pixels and the overlay
    % marker refer to the same physical location.
    detector_center_col = detector_center_xy(1);
    detector_center_row = detector_center_xy(2);
    row_idx = (detector_center_row-detector_half_size):(detector_center_row+detector_half_size);
    col_idx = (detector_center_col-detector_half_size):(detector_center_col+detector_half_size);
    if any(row_idx < 1) || any(row_idx > sy) || any(col_idx < 1) || any(col_idx > sx)
        error('Detector indices are outside simulation domain.');
    end
    detector_x_vals = x(col_idx);
    detector_y_vals = y(row_idx);
    detector_center_position = [mean(detector_x_vals), mean(detector_y_vals)];
    detector_rect = [detector_x_vals(1)-dx/2, detector_y_vals(end)-dy/2, numel(col_idx)*dx, numel(row_idx)*dy];

    % Transmission detector positioned behind the hydrogel
    % Transmission detector manually positioned behind the hydrogel
    transmission_center_xy = [480,210]; % [column, row] ordering
    transmission_center_col = transmission_center_xy(1);
    transmission_center_row = transmission_center_xy(2);
    transmission_row_idx = (transmission_center_row-detector_half_size):(transmission_center_row+detector_half_size);
    transmission_col_idx = (transmission_center_col-detector_half_size):(transmission_center_col+detector_half_size);
    if any(transmission_row_idx < 1) || any(transmission_row_idx > sy) || any(transmission_col_idx < 1) || any(transmission_col_idx > sx)
        error('Transmission detector indices are outside simulation domain.');
    end
    transmission_x_vals = x(transmission_col_idx);
    transmission_y_vals = y(transmission_row_idx);
    transmission_center_position = [mean(transmission_x_vals), mean(transmission_y_vals)];
    transmission_rect = [transmission_x_vals(1)-dx/2, transmission_y_vals(end)-dy/2, numel(transmission_col_idx)*dx, numel(transmission_row_idx)*dy];
    initial_average_samples = 3;
    final_average_samples = 5;
    wavelength_smoothing_window = 5;
    
    %% This is the field propagation loop
    ki=0;
    figure
    
    nt_end = 4000;
    % exposure_intensity = zeros(size(X));
    Ez_his = zeros(nt,1);
    GEz_his = gpuArray(Ez_his);
    transmission_intensity_his = zeros(nt,1);
    Gtransmission_intensity_his = gpuArray(transmission_intensity_his);
    source_intensity_his = zeros(nt,1);
    Gsource_intensity_his = gpuArray(source_intensity_his);
    peak = zeros(nt,2);
    Gpeak = gpuArray(peak);
    real_peak = [];
    Greal_peak = gpuArray(real_peak);
    reflection_wavelength = [];
    Greflection_wavelength = gpuArray(reflection_wavelength);
    detected_initial_wavelength = NaN;
    detected_final_wavelength = NaN;
    tic
    while ki<nt
        ki=ki+1;
        %Define source
        GJEz=Gsource*cos(omega*dt*ki);
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
        
        % GEz_his(ki,1) = GEz(100,400);
        % if (ki >= 2) && (GEz(100,400)^2 <= GEz_his(ki-1,1)^2) && (GEz_his(ki-1,1)^2 >= GEz_his(ki-2,1)^2)
        Gdetector_Ez = mean(GEz(row_idx,col_idx),'all');
        GEz_his(ki,1) = Gdetector_Ez;
        Gtransmission_patch = GEz(transmission_row_idx,transmission_col_idx);
        Gtransmission_intensity = mean(abs(Gtransmission_patch).^2,'all');
        Gtransmission_intensity_his(ki,1) = Gtransmission_intensity;
        Gsource_patch = Gsource(row_idx,col_idx)*cos(omega*dt*ki);
        Gsource_intensity = mean(abs(Gsource_patch).^2,'all');
        Gsource_intensity_his(ki,1) = Gsource_intensity;
        if (ki >= 3) && (GEz_his(ki,1)^2 <= GEz_his(ki-1,1)^2) && (GEz_his(ki-1,1)^2 >= GEz_his(ki-2,1)^2)
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
            transmission_intensity_his = gather(Gtransmission_intensity_his);
            peak = gather(Gpeak);
            real_peak = gather(Greal_peak);
            reflection_wavelength = gather(Greflection_wavelength);
            source_intensity_his = gather(Gsource_intensity_his);
            subplot(2,2,1);
            imagesc(x,y,Ez,[-1 1]/brightness)  
            hold on
            rectangle('Position',detector_rect,'EdgeColor','r','LineWidth',1.5)
            plot(detector_center_position(1),detector_center_position(2),'r+','LineWidth',1.5,'MarkerSize',8)
            rectangle('Position',transmission_rect,'EdgeColor',[0 0.6 0],'LineWidth',1.5)
            plot(transmission_center_position(1),transmission_center_position(2),'g+','LineWidth',1.5,'MarkerSize',8)
            axis image
            title(sprintf('TimeFrame %s',num2str(ki)))
            set(gca,'ydir','normal' )
            colormap(gca,emkc)
            alpha(1-obj1*0.1-obj2*0.2)
    
            subplot(2,2,2);
            plot(dt:dt:ki*dt,Ez_his(1:ki))
            xlabel('time (second)')
            ylabel('Ez Field')
            
            subplot(2,2,3);
            time_axis = dt:dt:ki*dt;
            source_intensity_slice = source_intensity_his(1:ki);
            reflection_intensity = Ez_his(1:ki).^2;
            transmission_intensity = transmission_intensity_his(1:ki);
            plot(time_axis,reflection_intensity,'DisplayName','Reflection region')
            hold on
             valid_peak_mask = peak(:,2)~=0;
            if any(valid_peak_mask)
                peak_times = peak(valid_peak_mask,1);
                peak_indices = max(1,min(ki,round(peak_times/dt)));
                peak_intensity = peak(valid_peak_mask,2).^2;
                plot(peak_times,peak_intensity,'DisplayName','Peaks')
            end
            plot(time_axis,transmission_intensity,'g','LineWidth',1.2,'DisplayName','Behind hydrogel (minus source)')
            xlabel('time (second)')
            ylabel('Intensity')
            legend('Location','best')
            hold off
    
            subplot(2,2,4);
            cla
            if ~isempty(reflection_wavelength)
                smoothing_window = min(wavelength_smoothing_window,numel(reflection_wavelength));
                smoothed_wavelength = movmean(reflection_wavelength,smoothing_window);
                plot(reflection_wavelength,'Color',[0.7 0.7 0.7],'DisplayName','Raw');
                hold on
                plot(smoothed_wavelength,'r','LineWidth',1.5,'DisplayName','Smoothed');
                initial_count = min(initial_average_samples,numel(reflection_wavelength));
                final_count = min(final_average_samples,numel(reflection_wavelength));
                detected_initial_wavelength = mean(reflection_wavelength(1:initial_count));
                detected_final_wavelength = mean(reflection_wavelength(end-final_count+1:end));
                yline(detected_initial_wavelength,'--','Initial avg','Color',[0 0.45 0.74]);
                yline(detected_final_wavelength,'--','Final avg','Color',[0.85 0.33 0.1]);
                legend('Location','best');
                title(sprintf('Final: %.1f nm',detected_final_wavelength*1e9));
                hold off
            else
                plot(nan,nan);
                title('Wavelength detection pending');
            end
            xlabel('Peak index')
            ylabel('wavelength (m)')
    
            drawnow
            if movvar==1
                frame = getframe(gcf);
                writeVideo(writerObj,frame);
            end
            tic
        end
    end
    reflection_wavelength = gather(Greflection_wavelength);
    reflection_wavelength = reflection_wavelength(isfinite(reflection_wavelength) & (reflection_wavelength>0));
    transmission_intensity_his = gather(Gtransmission_intensity_his);
    transmission_intensity_his = transmission_intensity_his(1:ki);
    source_intensity_his = gather(Gsource_intensity_his);
    source_intensity_his = source_intensity_his(1:ki);
    if isempty(reflection_wavelength)
        readout_wavelength(i) = NaN;
        readout_wavelength_initial(i) = NaN;
        readout_wavelength_final(i) = NaN;
    else
        initial_count = min(initial_average_samples,numel(reflection_wavelength));
        final_count = min(final_average_samples,numel(reflection_wavelength));
        detected_initial_wavelength = mean(reflection_wavelength(1:initial_count));
        detected_final_wavelength = mean(reflection_wavelength(end-final_count+1:end));
        readout_wavelength(i) = detected_final_wavelength;
        readout_wavelength_initial(i) = detected_initial_wavelength;
        readout_wavelength_final(i) = detected_final_wavelength;
        fprintf('Detected wavelength change: initial %.2f nm -> final %.2f nm\n',detected_initial_wavelength*1e9,detected_final_wavelength*1e9);
    end
    readout_peak_mask = peak(:,2)~=0;
    readout_intensity_temp = peak(readout_peak_mask,2).^2;
    if isempty(readout_intensity_temp)
        readout_intensity(i) = NaN;
    else
        peak_times = peak(readout_peak_mask,1);
        peak_indices = max(1,min(length(source_intensity_his),round(peak_times/dt)));
        peak_baseline = source_intensity_his(peak_indices);
        readout_intensity_temp = readout_intensity_temp - peak_baseline;
        readout_intensity(i) = readout_intensity_temp(end);
    end

    baseline_subtracted_transmission = transmission_intensity_his;
    finite_transmission = baseline_subtracted_transmission(isfinite(baseline_subtracted_transmission));
    if isempty(finite_transmission)
        readout_transmission_initial(i) = NaN;
        readout_transmission_final(i) = NaN;
    else
        initial_count = min(initial_average_samples,numel(finite_transmission));
        final_count = min(final_average_samples,numel(finite_transmission));
        readout_transmission_initial(i) = mean(finite_transmission(1:initial_count));
        readout_transmission_final(i) = mean(finite_transmission(end-final_count+1:end));
    end
    Ez_snapshot = gather(GEz);
    window_x = 0.5 - 0.5*cos(2*pi*(0:sx-1)/(sx-1));
    window_y = 0.5 - 0.5*cos(2*pi*(0:sy-1)/(sy-1));
    apodization = window_y.' * window_x;
    Ez_apodized = Ez_snapshot .* apodization;
    farfield_spectrum = fftshift(fft2(Ez_apodized));
    kx_vec = (-floor(sx/2):ceil(sx/2)-1)*(2*pi/lengthx);
    ky_vec = (-floor(sy/2):ceil(sy/2)-1)*(2*pi/lengthy);
    [KX,KY] = meshgrid(kx_vec,ky_vec);
    k_rho = sqrt(KX.^2 + KY.^2);
    k0 = 2*pi/incident_wavelength;
    propagating_mask = k_rho <= k0;
    if any(propagating_mask,"all")
        theta_values_deg = asind(min(1,k_rho(propagating_mask)/k0));
        spectral_intensity = abs(farfield_spectrum(propagating_mask)).^2;
        bin_idx = discretize(theta_values_deg,theta_bin_edges);
        valid_bins = ~isnan(bin_idx);
        if any(valid_bins)
            angular_profile = accumarray(bin_idx(valid_bins),spectral_intensity(valid_bins),[numel(theta_bin_centers),1],@mean,NaN);
            max_profile = max(angular_profile,[],'omitnan');
            if ~(isnan(max_profile) || max_profile==0)
                angular_profile = angular_profile./max_profile;
            end
            farfield_angular_intensity(i,:) = angular_profile;
        end
    end
    if i == length(wavelength)
        figure;
        imagesc(kx_vec/k0,ky_vec/k0,log10(abs(farfield_spectrum)+eps));
        axis image;
        set(gca,'YDir','normal');
        xlabel('k_x/k_0');
        ylabel('k_y/k_0');
        title('Log_{10} Far-field spectrum (final wavelength)');
        colorbar;
    end
end
%% Close video
if movvar==1
    close(writerObj);
end
figure
plot(readout_wavelength_initial*1e9,readout_intensity,'--','Color',[0 0.45 0.74],'DisplayName','Initial avg (minus source)')
hold on
plot(readout_wavelength_final*1e9,readout_intensity,'r-','DisplayName','Final avg (minus source)')

hold off
figure;
imagesc(theta_bin_centers, wavelength*1e9, farfield_angular_intensity);
set(gca,'YDir','normal');
xlabel('Observation angle (degrees)');
ylabel('Incident wavelength (nm)');
title('Normalized far-field angular intensity');
colorbar;
% xlabel('Wavelength (nm)')
% ylabel('Intensity')
% legend('Location','best')

figure
plot(wavelength*1e9,readout_transmission_initial,'--','Color',[0 0.6 0],'DisplayName','Initial avg behind hydrogel (minus source)')
hold on
plot(wavelength*1e9,readout_transmission_final,'-','Color',[0 0.45 0],'DisplayName','Final avg behind hydrogel (minus source)')
hold off
xlabel('Incident Wavelength (nm)')
ylabel('Transmitted Intensity')
legend('Location','best')
title('Transmission detector intensity behind hydrogel')
