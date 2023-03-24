clear all
close all


% PRO orbit parameters
rho_r   =  [50,64,78]; % m
rho_s   =  [0,0,0];   % m
rho_w   =  [0,60,140];  % m
alpha_r =  [deg2rad(90),deg2rad(90),deg2rad(90)];% rad
alpha_w =  [deg2rad(0),deg2rad(0),deg2rad(0)]; % rad 

%% ISS orbit Parameters 
mu_earth = 3.98e14; % m^3/s^2
ecc = 4.58e-4; %[-] eccentricity
p   = 6.80e6;  %[m] semimajor axis
angular_momentum = sqrt(p*mu_earth*(1-ecc^2)); %[m2/s] specific angular momenum
inc = deg2rad(55.99) ;% [rad] orbit inclination
mean_motion = angular_momentum/p^2;
period = 2*pi*sqrt(p^3/mu_earth);

% ISS plot
ISS_model = stlread('ISS_2016.stl');
% PRO = readtable(fullfile(simulationos_data_directory,'PRO_visualization.csv'));
% resize and rotate ISS rotation
angle1 = pi/2
R1 = [cos(angle1) ,-sin(angle1) ,0;
     sin(angle1),cos(angle1) ,0;
     0         , 0         ,1];
angle2 = -pi

R2 = [cos(angle2), 0 ,sin(angle2);
     0          , 1 ,0;
     -sin(angle2) , 0 ,cos(angle2)];

iss_max_one_side_length = 54.5    ; % real size of the ISS along z
z_max         = max(ISS_model.Points(:,3)); % model size z
stretch_ratio = iss_max_one_side_length/z_max
S             = stretch_ratio * eye(3); % stratch matrix
Transform     = R2*R1*S;

ISS_model_rotaterd = triangulation(ISS_model.ConnectivityList,(Transform*ISS_model.Points')');


% ISS PLANT

figure
t = tiledlayout(6,6);
ax=nexttile(1,[6 2]);
hold on
trimesh(ISS_model_rotaterd,"EdgeColor","red","EdgeAlpha",0.3,'FaceColor','red','FaceAlpha',0.2)
xlabel("radial [m]")
ylabel("along-track [m]")
zlabel("cross-track [m]")
view(0,0)

ax = nexttile(15,[4 4]);
hold on
trimesh(ISS_model_rotaterd,"EdgeColor","red","EdgeAlpha",0.3,'FaceColor','red','FaceAlpha',0.2)
xlabel("radial [m]")
ylabel("along-track [m]")
zlabel("cross-track [m]")
view(-90,0)


ax = nexttile(3,[2 4]);
hold on
view(ax,-90,90)
trimesh(ISS_model_rotaterd,"EdgeColor","red","EdgeAlpha",0.3,'FaceColor','red','FaceAlpha',0.2)
xlabel("radial [m]")
ylabel("along-track [m]")
zlabel("cross-track [m]")

%ISS with PRO
time = linspace(0,period,200)
figure
ax =gca();
hold on
trimesh(ISS_model_rotaterd,"EdgeColor","red","EdgeAlpha",0.3,'FaceColor','red','FaceAlpha',0.2)
for jj = 1:length(rho_w)
    % PRO anonymus function
    PRO_position     = @(t)                 [rho_r(jj)*sin(mean_motion*t + alpha_r(jj)) ; rho_s(jj) + 2*rho_r(jj)*cos(mean_motion*t + alpha_r(jj));rho_w(jj)*sin(mean_motion*t + alpha_w(jj))];
    trajectory = PRO_position(time)
    plot3(trajectory(1,:),trajectory(2,:),trajectory(3,:),"blue")
end
view(30,60)
xlabel("radial [m]")
ylabel("along-track [m]")
zlabel("cross-track [m]")





