clear all
close all


path_list = ["./agent0","./agent1","./agent2"];
ax_list = cell(1,5);

cbf_fig = figure();
cbf_fig_ax = gca();

ax_list2 = cell(1,4);
ax_list3 = cell(1,3);

for jj = 1:4
    ax_list{jj} = subplot(4,1,jj);
    hold(ax_list{jj})
    
end

control_figure = figure;
control_figure_ax = gca();

for jj = 1:4
    ax_list2{jj} = subplot(4,1,jj);
    hold(ax_list2{jj})
end


angle_figure = figure;
angle_figure_ax = gca();

for jj = 1:3
    ax_list3{jj} = subplot(3,1,jj);
    hold(ax_list3{jj})
end

%%


for kk = 1:length(path_list) 

    CBF         = readtable(fullfile(path_list(kk),'CBF_data_frame.csv'));
    control     = readtable(fullfile(path_list(kk),'control_data_frame.csv'));
    state       = readtable(fullfile(path_list(kk),'state_data_frame.csv'));
    state_ref   = readtable(fullfile(path_list(kk),'reference_state_data_frame.csv'));
    lwidth      = 2;
    CBF.time    = CBF.time/60 ;

    % plot the CBF performace
    
    % velocity CBF
   
    plot(ax_list{1},CBF.time,CBF.CBF_velocity,LineWidth=lwidth,DisplayName=['inspector ',num2str(kk)]);
    ax_list{1}.XLabel.String = "time [min]";
    ax_list{1}.YLabel.String = "h_{\delta v}";
    ax_list{1}.YLim = [-0.03,max(CBF.CBF_velocity)*1.2];
    % position CBF
    
    plot(ax_list{2},CBF.time,CBF.CBF_position,LineWidth=lwidth);
    ax_list{2}.XLabel.String = "time [min]";
    ax_list{2}.YLabel.String = "h_{\delta r}";
    ax_list{2}.YLim = [-0.03,max(CBF.CBF_position)*1.2];
    % position HOBF
    
    plot(ax_list{3},CBF.time,CBF.velocity_constraint,LineWidth=lwidth);
    ax_list{3}.XLabel.String = "time [min]";
    ax_list{3}.YLabel.String = "\psi_{\delta v}";
    ax_list{3}.YLim = [-0.003,max(CBF.velocity_constraint)*1.2];
    % state performace
    state_array           = table2array(state);
    reference_state_array = table2array(state_ref);
    state_error = state_array-reference_state_array;
    
    pos_error = vecnorm(state_error(:,1:3)');
    vel_error = vecnorm(state_error(:,4:end)');

    plot(ax_list{4},CBF.time,CBF.position_constraint,LineWidth=lwidth);
    ax_list{4}.XLabel.String = "time [min]";
    ax_list{4}.YLabel.String = "\zeta_{\delta r}";
    ax_list{4}.YLim = [-0.03,max(CBF.position_constraint)*1.2];

%     plot(ax_list{5},CBF.time,vel_error,LineWidth=lwidth);
%     ax_list{5}.XLabel.String = "time [min]";
%     ax_list{5}.YLabel.String = "\|e_{\delta v}\|";

%     control performance

    stairs(ax_list2{1},CBF.time,control.u_x,LineWidth=lwidth);
    ax_list2{1}.XLabel.String = "time [min]";
    ax_list2{1}.YLabel.String = "u_r";
    

    stairs(ax_list2{2},CBF.time,control.u_y,LineWidth=lwidth);
    ax_list2{2}.XLabel.String = "time [min]";
    ax_list2{2}.YLabel.String = "u_s";

    stairs(ax_list2{3},CBF.time,control.u_z,LineWidth=lwidth);
    ax_list2{3}.XLabel.String = "time [min]";
    ax_list2{3}.YLabel.String = "u_w";

    plot(ax_list2{4},CBF.time,CBF.state_history,LineWidth=lwidth);
    ax_list2{4}.XLabel.String = "time [min]";
    ax_list2{4}.YLabel.String = "discrete state";
    ax_list2{4}.YTick = [0,1];
    ax_list2{4}.YTickLabel = ["a","n/a"];


    control_array = table2array(control);
    pos_state_error   = state_error(:,1:3);
    vel_state_error   = state_error(:,4:end);
    
    dot_pos_control   = sum(pos_state_error.*control_array,2)./vecnorm(pos_state_error')'./vecnorm(control_array')';
    dot_vel_control   = sum(vel_state_error.*control_array,2)./vecnorm(vel_state_error')'./vecnorm(control_array')';
    dot_pos_vel   = sum(pos_state_error.*vel_state_error,2)./vecnorm(vel_state_error')'./vecnorm(pos_state_error')';

    plot(ax_list3{1},CBF.time,acosd(dot_pos_control),LineWidth=lwidth);
    ax_list3{1}.XLabel.String = "time [min]";
    ax_list3{1}.YLabel.String = "dot pos control";
    
    plot(ax_list3{2},CBF.time,acosd(dot_vel_control),LineWidth=lwidth);
    ax_list3{2}.XLabel.String = "time [min]";
    ax_list3{2}.YLabel.String = "dot vel control  ";

     plot(ax_list3{3},CBF.time,acosd(dot_pos_vel),LineWidth=lwidth);
    ax_list3{3}.XLabel.String = "time [min]";
    ax_list3{3}.YLabel.String = "dot pos vel ";


end

legend(ax_list{1})

