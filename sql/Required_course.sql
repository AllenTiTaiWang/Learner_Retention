select pa.user_id, pa.application_id, pas.status_id, pp.name as payment_plan, rep.response, (extract(epoch from sum(age(us.session_end, us.session_start))/3600)::integer) as hours_online, p.program_name, ap.type_name as application_type_name,
pa.professional_assoc, 
ud.referrer, ud.practice_type, pch.state as home_state, pcw.state as work_state, ud.gender, ud.home_country, ud.work_country, ud.job_title, pc.short_title as curriculum_id
from program_application pa
join program p on p.program_id = pa.program_id
join program_application_status pas on pas.status_id = pa.status
join program_curriculum pc on pc.program_id = pa.program_id
join application_type ap on ap.application_type_id=pa.application_type_id
join application_status_lookup s on s.status = pa.status
join curriculum_requirement cr on cr.curriculum_id = pc.curriculum_id
join course_offering co on co.requirement_id = cr.requirement_id
join course_enrollment ce on ce.offering_id = co.offering_id 
join user_data ud on ud.user_id = pa.user_id
join user_session us on us.user_id = pa.user_id

join (select pa.user_id, pc.short_title, count(ur.text_answer) as response
        from program_application pa
        join program p on p.program_id = pa.program_id
        join program_curriculum pc on pc.program_id = pa.program_id
        join curriculum_requirement cr on cr.curriculum_id = pc.curriculum_id
		join course_offering co on co.requirement_id = cr.requirement_id
        join user_module_activity uma on uma.user_id = pa.user_id and uma.offering_id = co.offering_id       
        join user_response ur on uma.activity_id = ur.activity_id 
        where char_length(ur.text_answer) > 10 and p.program_type = 'IHeLp' and p.program_name != 'IHeLp'
        group by pa.user_id, pc,short_title)rep on rep.user_id = pa.user_id and rep.short_title = pc.short_title


left join payment_plan pp on pp.payment_plan_id=pa.payment_plan_id
left outer join postal_codes pch on pch.postal_code = substr(ud.home_postal_code, 0, 6)
left outer join postal_codes pcw on pcw.postal_code = substr(ud.work_postal_code, 0, 6)
where p.program_type = 'IHeLp' and p.program_name != 'IHeLp' and 
pas.status_id in ('W','G','D','I') and ud.is_test_user = false
group by pa.user_id, pa.application_id, pas.status_id,payment_plan, rep.response, p.program_name, application_type_name,
pa.professional_assoc, 
ud.referrer, ud.practice_type, pch.state, pcw.state, ud.gender, ud.home_country, ud.work_country, ud.job_title, pc.short_title
