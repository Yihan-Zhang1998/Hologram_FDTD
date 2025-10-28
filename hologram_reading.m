%% TM mode simulation
% Always utilize SI units
% Reading
% Whitelight source simulation.
% hologram_recording
close all

gpuDeviceTable
GPU1 = gpuDevice(1);
%% Import figures
msource = imresize(imread('reading_source.png'),[570,570]);
shelter = imresize(imread('source_part.png'),[570,570]);
lens = imresize(imread('lens.png'),[570,570]);
%% Initialize video
movvar=1;
fps=30;
if movvar==1
    writerObj = VideoWriter('20251028_flat_mirror_reading_Gaussian_FullWavelength.mp4','MPEG-4' );
    writerObj.FrameRate = fps;
    open(writerObj);
end

%% Universal constants
epsilon0=8.854187817e-12;
mu0=12.566370e-7;
c=1/sqrt(epsilon0*mu0);

%% Plot definitions
brightness=0.05;  %brightness of plot
nt=1500; %total number os time steps
waitdisp=15; %wait to display

lengthx=5e-6;  %size of the image in x axis SI
lengthy=lengthx;  %size of the image in y axis SI

[sy,sx]=size(msource(:,:,1));
%% Gaussian source definition (continuous-wave beam)
gaussian_beam_center = [-0.35*lengthx, 0]; % [x0, y0] in metres
gaussian_beam_waist = 0.45e-6;            % 1/e field radius (metres)
background_index_for_source = 1.37;       % Launch medium index for phase ramp

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
wavelength = linspace(300e-9,700e-9,4);
wavelength = [wavelength, 355e-9];
wavelength = sort(wavelength);%linspace(295e-9,565e-9,10);
readout_wavelength_initial = zeros(length(wavelength),1);
readout_wavelength_final = zeros(length(wavelength),1);
readout_intensity = zeros(length(wavelength),1);
readout_transmission_initial = zeros(length(wavelength),1);
readout_transmission_final = zeros(length(wavelength),1);
theta_bin_edges = linspace(0,180,181);
theta_bin_centers = (theta_bin_edges(1:end-1)+theta_bin_edges(2:end))/2;
% Angle bins measured relative to the -x axis.
farfield_angular_intensity = NaN(length(wavelength),numel(theta_bin_centers));
farfield_angular_sample_counts = zeros(length(wavelength),numel(theta_bin_centers));
farfield_spectrum_log = NaN(sy,sx,length(wavelength));
kx_normalized = cell(length(wavelength),1);
ky_normalized = cell(length(wavelength),1);
reflection_halfspace_energy = NaN(length(wavelength),1);
transmission_halfspace_energy = NaN(length(wavelength),1);
reflection_halfspace_energy_normalized = NaN(length(wavelength),1);
transmission_halfspace_energy_normalized = NaN(length(wavelength),1);
lens_refractive_index_spectrum = NaN(length(wavelength),1);
propagation_angle_deg = NaN(length(wavelength),1);
dominant_halfspace = repmat({''},length(wavelength),1);
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
    radial_sq = (X - gaussian_beam_center(1)).^2 + (Y - gaussian_beam_center(2)).^2;
    gaussian_profile = exp(-radial_sq/(gaussian_beam_waist^2));
    gaussian_profile = gaussian_profile./max(gaussian_profile,[],"all");
    source = gaussian_profile;
    Gsource = gpuArray(source);
    GX = gpuArray(X);
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
    % obj3 = shelter;
    % obj3 = round(double(obj3(:,:,3))/255);
    obj3 = zeros(570,570);
    % obj4 = lens;
    % obj4 = round(double(obj4(:,:,1))/255);
    obj4 = zeros(570,570);
    %imagesc(x,y,obj2);

    RI0=1.34;  %Background refractive index
    alpha1=log(2)/100e-9;  %Absorbtion coeficient with half loss in 100 nm
    RI1=1+1i*c*alpha1/2/omega(1); %Refractive indexwith absorption coeficient
    RI2=1.34;    %Refractive index of P1
    RI3=1.67;  %Refractive index of P2
    RI4=10+1i*5;
    RI_lens = computeLensRefractiveIndex(incident_wavelength);
    lens_refractive_index_spectrum(i) = RI_lens;
    
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
    detector_center_xy = [160,450]; % [column (x-index), row (y-index)] as referenced in plots
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
    transmission_center_xy = [500,285]; % [column, row] ordering
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
    % Pre-compute the spatial phase ramp that launches the Gaussian toward +x
    free_space_wavenumber = background_index_for_source * (2*pi/incident_wavelength);
    Gsource_phase = free_space_wavenumber .* (GX - gaussian_beam_center(1));
    while ki<nt
        ki=ki+1;
        %Define source
        GJEz = Gsource .* cos(omega*dt*ki - Gsource_phase); % exp(-((t - t0) / width)^2) * sin(2 * pi * f0 * (t - t0));
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
        local_phase = Gsource_phase(row_idx,col_idx);
        Gsource_patch = Gsource(row_idx,col_idx) .* cos(omega*dt*ki - local_phase);
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
            plot(time_axis,transmission_intensity,'g','LineWidth',1.2,'DisplayName','Behind hydrogel')
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
                % if ~isopen(writerObj)
                %     open(writerObj);
                % end
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
    spectral_intensity_map = abs(farfield_spectrum).^2;
    farfield_spectrum_log(:,:,i) = log10(abs(farfield_spectrum)+eps);
    kx_normalized{i} = kx_vec/k0;
    ky_normalized{i} = ky_vec/k0;
    if any(propagating_mask,"all")
        reflection_mask = propagating_mask & (KX < 0);
        transmission_mask = propagating_mask & (KX > 0);
        reflection_energy = sum(spectral_intensity_map(reflection_mask),"all");
        transmission_energy = sum(spectral_intensity_map(transmission_mask),"all");
        total_propagating_energy = reflection_energy + transmission_energy;
        reflection_halfspace_energy(i) = reflection_energy;
        transmission_halfspace_energy(i) = transmission_energy;
        if total_propagating_energy > 0
            reflection_halfspace_energy_normalized(i) = reflection_energy/total_propagating_energy;
            transmission_halfspace_energy_normalized(i) = transmission_energy/total_propagating_energy;
        end

        dominant_mask = reflection_mask;
        dominant_halfspace{i} = 'reflection';
        if transmission_energy > reflection_energy
            dominant_mask = transmission_mask;
            dominant_halfspace{i} = 'transmission';
        end

        propagation_kx = KX(dominant_mask);
        propagation_ky = KY(dominant_mask);
        spectral_intensity_selected = spectral_intensity_map(dominant_mask);
        direction_magnitude = hypot(propagation_kx,propagation_ky);
        valid_direction = direction_magnitude > 0;
        propagation_kx = propagation_kx(valid_direction);
        propagation_ky = propagation_ky(valid_direction);
        spectral_intensity_selected = spectral_intensity_selected(valid_direction);
        direction_magnitude = direction_magnitude(valid_direction);

        if ~isempty(spectral_intensity_selected)
            cos_angles = max(-1,min(1,-propagation_kx./direction_magnitude));
            theta_values_deg = real(acosd(cos_angles));
            bin_idx = discretize(theta_values_deg,theta_bin_edges);
            valid_bins = ~isnan(bin_idx);
            if any(valid_bins)
                angular_profile = accumarray(bin_idx(valid_bins),spectral_intensity_selected(valid_bins),[numel(theta_bin_centers),1],@mean,NaN);
                max_profile = max(angular_profile,[],'omitnan');
                if ~(isnan(max_profile) || max_profile==0)
                    angular_profile = angular_profile./max_profile;
                end
                farfield_angular_intensity(i,:) = angular_profile;

                energy_sum = sum(spectral_intensity_selected(valid_bins));
                if energy_sum > 0
                    propagation_angle_deg(i) = sum(theta_values_deg(valid_bins).*spectral_intensity_selected(valid_bins))/energy_sum;
                end
            end
            % farfield_angular_intensity(i,:) = angular_profile;
            % farfield_angular_sample_counts(i,:) = angular_counts;
            % empty_bin_count = nnz(~populated);
            % if empty_bin_count > 0
            %     fprintf(['Wavelength %.0f nm: %d of %d polar bins received no propagating ', ...
            %         'Fourier samples (grid directions between discrete k-points).\n'], ...
            %         incident_wavelength*1e9, empty_bin_count, numel(theta_bin_centers));
            % end
        end
    end
    % save(sprintf("hologram_reading_lambda_%s.mat",wavelength(i)))
    % open(writerObj);
end
%% Close video
if movvar==1
    close(writerObj);
end
figure
% plot(readout_wavelength_initial*1e9,readout_intensity,'--','Color',[0 0.45 0.74],'DisplayName','Initial avg (minus source)')
% hold on
plot(wavelength*1e9,readout_intensity,'r-','DisplayName','Final avg (minus source)')
hold off
drawnow

figure;
imagesc(theta_bin_centers, wavelength*1e9, farfield_angular_intensity.*max_profile);
set(gca,'YDir','normal');
xlabel('Angle between propagation direction and -x axis (degrees)');
ylabel('Incident wavelength (nm)');
title('Far-field angular intensity');
colorbar;
drawnow

figure;
imagesc(theta_bin_centers, wavelength*1e9, farfield_angular_sample_counts);
set(gca,'YDir','normal');
xlabel('Polar angle from +x toward +y (degrees)');
ylabel('Incident wavelength (nm)');
title('Count of propagating Fourier samples per polar bin');
colorbar;

drawnow

%% Visualize angular routing for each wavelength
valid_profiles = ~all(isnan(farfield_angular_intensity),2);
if any(valid_profiles)
    figure;
    pax = polaraxes;
    hold(pax,'on');
    peak_angles = NaN(length(wavelength),1);
    peak_values = NaN(length(wavelength),1);
    for i_plot = 1:length(wavelength)
        profile = farfield_angular_intensity(i_plot,:);
        if all(isnan(profile))
            continue;
        end
        angle_rad = deg2rad(theta_bin_centers);
        rgb = wavelength_to_rgb(wavelength(i_plot)*1e9);
        polarplot(pax, angle_rad, profile, 'Color', rgb, ...
            'DisplayName', sprintf('%.0f nm', wavelength(i_plot)*1e9));
        [peak_values(i_plot), peak_idx] = max(profile);
        if ~isnan(peak_values(i_plot)) && peak_values(i_plot) > 0
            peak_angles(i_plot) = theta_bin_centers(peak_idx);
            polarscatter(pax, deg2rad(peak_angles(i_plot)), peak_values(i_plot), 36, rgb, 'filled');
        end
    end
    title(pax, 'Angular distribution by wavelength');
    legend(pax, 'Location', 'eastoutside');
    hold(pax,'off');

    drawnow

    finite_peaks = ~isnan(peak_angles);
    if any(finite_peaks)
        figure;
        plot(wavelength(finite_peaks)*1e9, peak_angles(finite_peaks), 'o-');
        xlabel('Incident wavelength (nm)');
        ylabel('Peak intensity angle (degrees from -x toward +y)');
        title('Dominant emission angle versus wavelength');
        grid on;
        drawnow

    end
end

% xlabel('Wavelength (nm)')
% ylabel('Intensity')
% legend('Location','best')
figure;
hold on;
reflection_idx = strcmp(dominant_halfspace,'reflection') & ~isnan(propagation_angle_deg);
transmission_idx = strcmp(dominant_halfspace,'transmission') & ~isnan(propagation_angle_deg);
if any(reflection_idx)
    plot(wavelength(reflection_idx)*1e9,propagation_angle_deg(reflection_idx),'-o', ...
        'Color',[0 0.45 0.74],'DisplayName','Reflection half-space');
end
if any(transmission_idx)
    plot(wavelength(transmission_idx)*1e9,propagation_angle_deg(transmission_idx),'-s', ...
        'Color',[0.85 0.33 0.1],'DisplayName','Transmission half-space');
end
hold off;
xlabel('Incident wavelength (nm)');
ylabel('Dominant angle to -x axis (degrees)');
title('Dominant propagation angle relative to -x axis');
legend('Location','best');
grid on;
for i_plot = 1:length(wavelength)
    spectrum_log = farfield_spectrum_log(:,:,i_plot);
    if any(isfinite(spectrum_log),'all')
        figure;
        imagesc(kx_normalized{i_plot}, ky_normalized{i_plot}, spectrum_log);
        axis image;
        set(gca,'YDir','normal');
        xlabel('k_x/k_0');
        ylabel('k_y/k_0');
        title(sprintf('far-field spectrum (%.0f nm)', wavelength(i_plot)*1e9));
        colorbar;
        drawnow

    end
end

figure;
plot(wavelength*1e9,lens_refractive_index_spectrum,'-d','Color',[0.2 0.2 0.7]);
xlabel('Incident Wavelength (nm)');
ylabel('Lens refractive index');
title('Dispersive lens material model');
drawnow

figure
plot(wavelength*1e9,readout_transmission_initial,'--','Color',[0 0.6 0],'DisplayName','Initial avg behind hydrogel (minus source)')
hold on
plot(wavelength*1e9,readout_transmission_final,'-','Color',[0 0.45 0],'DisplayName','Final avg behind hydrogel (minus source)')
hold off
xlabel('Incident Wavelength (nm)')
ylabel('Transmitted Intensity')
legend('Location','best')
title('Transmission detector intensity behind hydrogel')
drawnow

save("hologram_reading.mat")
% broadband_light

function rgb = wavelength_to_rgb(lambda_nm)
%wavelength_to_rgb Convert visible wavelength to approximate RGB triple.
%   lambda_nm: scalar wavelength in nanometers.
%   Returns RGB values in the range [0,1].

    if lambda_nm < 380 || lambda_nm > 780
        rgb = [0, 0, 0];
        return;
    end

    if lambda_nm < 440
        attenuation = (lambda_nm - 380) / (440 - 380);
        R = -(lambda_nm - 440) / (440 - 380);
        G = 0.0;
        B = 1.0;
    elseif lambda_nm < 490
        attenuation = 1.0;
        R = 0.0;
        G = (lambda_nm - 440) / (490 - 440);
        B = 1.0;
    elseif lambda_nm < 510
        attenuation = 1.0;
        R = 0.0;
        G = 1.0;
        B = -(lambda_nm - 510) / (510 - 490);
    elseif lambda_nm < 580
        attenuation = 1.0;
        R = (lambda_nm - 510) / (580 - 510);
        G = 1.0;
        B = 0.0;
    elseif lambda_nm < 645
        attenuation = 1.0;
        R = 1.0;
        G = -(lambda_nm - 645) / (645 - 580);
        B = 0.0;
    else
        attenuation = exp(-0.0015 * (lambda_nm - 645));
        R = 1.0;
        G = 0.0;
        B = 0.0;
    end

    gamma = 0.8;
    rgb = [R, G, B] .* attenuation;
    rgb = max(rgb, 0);
    rgb = rgb .^ gamma;
end

function n = computeLensRefractiveIndex(lambda_meters)
%COMPUTELENSREFRACTIVEINDEX Dispersion model for the hologram lens material.
%   Uses the BK7 Sellmeier equation parameterization to return the real
%   refractive index as a function of wavelength. The input wavelength is in
%   meters and the output index is dimensionless.

    lambda_um = lambda_meters * 1e6; % Convert to micrometers for Sellmeier coefficients
    lambda_um_sq = lambda_um.^2;

    % Sellmeier coefficients for a borosilicate (BK7-like) glass
    B1 = 1.03961212;
    B2 = 0.231792344;
    B3 = 1.01046945;
    C1 = 0.00600069867;
    C2 = 0.0200179144;
    C3 = 103.560653;

    n_squared = 1 + (B1 .* lambda_um_sq) ./ (lambda_um_sq - C1) ...
                 + (B2 .* lambda_um_sq) ./ (lambda_um_sq - C2) ...
                 + (B3 .* lambda_um_sq) ./ (lambda_um_sq - C3);
    n_squared = max(real(n_squared), 0);

    n = sqrt(n_squared);
end
