#-------------------------------------------------------------------------------
#
# Copyright (c) 2013
# IMB, RWTH Aachen University,
# ISM, Brno University of Technology
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in the AramisCDT top directory "license.txt" and may be
# redistributed only under the conditions described in the aforementioned
# license.
#
# Thanks for using Simvisage open source!
#
#-------------------------------------------------------------------------------
from traits.api import HasTraits

class AramisMixIn(HasTraits):
    '''Mixin class for clients describing a particular test run.
    It captures the workflow involved in the analysis of a test
    using the aramis_cdt functionality:

    1. Link the experiment with the aramis data including the specification
       of a resolution.

    2. Return processed data like, the positions and number of cracks

    3. Plot the results of the analysis into a supplied 2D or 3D toolkits
       (pylab, mlab)
    '''

    #===========================================================================
    # 2D-ARAMIS PROCESSING
    #===========================================================================

    aramis_resolution_key = Str('Xf15s3-Yf15s3')
    '''Specification of the resolution of the measured aramis field
    '''

    start_t_aramis = Float(5.0)
    '''Start time of aramis measurement.
    '''

    delta_t_aramis = Float(5.0)
    '''Delta between aramis snapshots.
    '''

    n_steps_aramis = Property(Int)
    def _get_n_steps_aramis(self):
        return self.aramis_info.number_of_steps

    t_aramis = Property(Array('float'), depends_on='data_file, start_t, delta_t, aramis_resolution_key')
    @cached_property
    def _get_t_aramis(self):
        start_t = self.start_t_aramis
        delta_t = self.delta_t_aramis
        n_steps = self.n_steps_aramis
        t_max = self.t[self.w_cut_idx]
        # print 'n_steps', n_steps
        # print 't_max', t_max
        t_aramis_full_range = np.linspace(start_t, n_steps * delta_t, n_steps)
        # print 't-aramis_full_range', t_aramis_full_range
        # print 't_aramis_full_range[t_aramis_full_range < t_max]', t_aramis_full_range[t_aramis_full_range < t_max]
        return t_aramis_full_range[t_aramis_full_range < t_max]

    # TO-DO: def some method to cut the time at a variable position, depending on strains at the end
    # cut the last values from t_aramis,because of very high strains at the end due to failure
    t_aramis_cut = Property(Array('float'), depends_on='data_file, start_t, delta_t, aramis_resolution_key')
    @cached_property
    def _get_t_aramis_cut(self):
        return self.t_aramis[ :-2]

    n_steps = Property
    @cached_property
    def _get_n_steps(self):
        'number of time steps in aramis after limiting the steps with t_max'
        # print 'n_steps', len(self.t_aramis)
        return len(self.t_aramis_cut)

    aramis_info = Property(depends_on='data_file,aramis_resolution_key')
    @cached_property
    def _get_aramis_info(self):
        af = self.get_cached_aramis_file(self.aramis_resolution_key)
        if af == None:
            return None
        return AramisInfo(data_dir=af)

    #--------------------------------------------------------------------------------
    # crack bridge strain
    #--------------------------------------------------------------------------------

    crack_filter_avg = Property(depends_on='data_file,aramis_resolution_key')
    @cached_property
    def _get_crack_filter_avg(self):
        ''' method to get number and position of cracks
        '''

        ai = self.aramis_info
        if ai == None:
            return None
        ad = AramisData(aramis_info=self.aramis_info,
                        evaluated_step_idx=self.n_steps)
        absa = AramisCDT(aramis_info=self.aramis_info,
                         crack_detect_idx=self.n_steps,
                         aramis_data=ad,
                         integ_radius=5,
                         ddd_ux_avg_threshold=-1e-4,
                         ddd_ux_threshold=-1e-4)

        print 'crack detect step', absa.crack_detect_idx
        print 'testing n_cracks', absa.number_of_cracks_avg
        # print 'ad.x_arr_undeformed', ad.x_arr_undeformed [0]
        print 'ad.x_arr_undeformed[0, absa.crack_filter_avg]', ad.x_arr_undeformed[0, absa.crack_filter_avg]
        # print 'ad.length_x_undeformed', ad.length_x_undeformed
        # print 'ad.length_y_undeformed', ad.length_y_undeformed
        # print absa.crack_filter_avg
        return ad.x_arr_undeformed[0, absa.crack_filter_avg]

    def _plot3d_ddd_ux(self, mlab):
        pass
