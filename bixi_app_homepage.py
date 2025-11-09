import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
import pydeck as pdk

def generate_colour_dot_plot(df, station_name, weekday, bike_or_dock):
    num_of_weeks_observed = len(df[df['weekday']==weekday][df[df['weekday']==weekday]['station']==station_name])
    ex_station = df[df['station'] == station_name]
    ex_station = ex_station[ex_station['weekday'] == weekday]
    ex_station['rides_departure'] = ex_station['rides_departure']/num_of_weeks_observed
    ex_station['rides_arrival'] = ex_station['rides_arrival']/num_of_weeks_observed
    ex_station['net_rides'] = ex_station['rides_departure'] - ex_station['rides_arrival'] if bike_or_dock=='🅿️' else ex_station['rides_arrival'] - ex_station['rides_departure']

    # Define your color rules
    def get_color(v):
        if (v>=-0.5) & (v<=0.5):
            return "orange"
        elif v == 676767:
            return 'white'
        elif v > 0.5:
            return "green"
        else:
            return "red"
        
    colour_dot_plot_df = ex_station.copy()

    if len(ex_station) != 24:
        full_hours = pd.DataFrame({'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]})
        colour_dot_plot_df = full_hours.merge(colour_dot_plot_df, on='hour', how='left')

    # Fill missing hours with defaults
    colour_dot_plot_df['value'] = colour_dot_plot_df['net_rides'].fillna(676767)

    colour_dot_plot_df["color"] = colour_dot_plot_df["value"].apply(get_color)


    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        y=[0]*len(colour_dot_plot_df),
        mode="markers+text",
        marker=dict(
            size=18,
            color=colour_dot_plot_df["color"],
            line=dict(width=0)
        ),
        text=[f"{h}h" for h in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]],
        textposition="top center",
        hoverinfo='skip'
    ))

    # Remove axes and grid
    fig.update_layout(
        height=150,
        width=1300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig



def generate_dist_plot(df, station_name, weekday, bike_or_dock, language):
    num_of_weeks_observed = len(df[df['weekday']==weekday][df[df['weekday']==weekday]['station']==station_name])
    ex_station = df[df['station'] == station_name]
    ex_station = ex_station[ex_station['weekday'] == weekday]
    ex_station['rides_departure'] = ex_station['rides_departure']/num_of_weeks_observed
    ex_station['rides_arrival'] = ex_station['rides_arrival']/num_of_weeks_observed
    ex_station['net_rides'] = ex_station['rides_departure'] - ex_station['rides_arrival'] if bike_or_dock=='🅿️' else ex_station['rides_arrival'] - ex_station['rides_departure']

    # Create figure
    fig = go.Figure()

    # Add first line (departures)
    fig.add_trace(go.Scatter(
        x=ex_station['hour'], y=ex_station['rides_departure'],
        mode='lines',
        name='Rides departure',
        fill='tozeroy',  # Fill to bottom
        line=dict(color='green' if bike_or_dock=='🅿️' else 'orange', width=2)
    ))

    # Add second line (arrivals)
    fig.add_trace(go.Scatter(
        x=ex_station['hour'], y=ex_station['rides_arrival'],
        mode='lines',
        name='Rides arrival',
        fill='tozeroy',
        line=dict(color='orange' if bike_or_dock=='🅿️' else 'green', width=2)
    ))

    # Customize layout
    fig.update_layout(
        title="",
        xaxis_title="Hour of Day" if language=='EN' else 'Heure de la journée',
        yaxis_title="Ride Count" if language=='EN' else "Nombre de trajets",
        xaxis=dict(tickmode='linear', dtick=1),
        template='simple_white',
        showlegend=False,
        height=300,
        width=1200
    )
    return fig


# ======================================
# 🏙️ APP CONFIG
# ======================================
st.set_page_config(
    page_title="Bixi Station Usage Explorer – Montréal",
    page_icon="🚲",
    layout="wide"
)

# ======================================
# 🌐 LANGUAGE SETUP & TABS SETUP
# ======================================
if "lang" not in st.session_state:
    st.session_state.lang = "FR"

language_homepage = st.segmented_control(
        "", ['EN', 'FR'], selection_mode="single", default=st.session_state.lang
    )
if language_homepage=='EN':
    st.session_state.lang = "EN"
if language_homepage=='FR':
    st.session_state.lang = "FR"

texts = {
    'EN' : {
        'tab_home' : 'Home',
        'tab_stations' : 'Explore Stations',
        'tab_bixi_wrap' : 'Bixi Wrap',
        'all' : 'All',
        'departure' : 'departures',
        'arrival' : 'arrivals',
        'title_homepage' : 'Decoding Bixi Rides Pattern',
        'intro_homepage' : "Discover when each Bixi station fills up or empties for any given day. Perfect for planning your rides or understanding the city’s biking rhythm.",
        'intro_stations' : "Find out about your station's rythm and plan your trips smarter",
        'station_selection_stations' : "Pick a station and start exploring",
        'day_selection_stations' : "Pick a day of the week:",
        'list_of_day_selections_stations' : ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'bike_dock_selection_stations' : "Looking for a bike or a dock",
        'station_not_found_recently' : "This station has not been found/used for some time. It is possible that the station does not exist anymore or has been moved.",
        'station_stats_departure_stations' : "Most departures occur at:",
        'station_stats_arrival_stations' : "Most arrivals occur at:",
        'colour_plot_legend_green_dock_stations' : "🟢: Good chance you'll find a dock",
        'colour_plot_legend_green_bike_stations' : "🟢: Good chance you'll find a bike",
        'colour_plot_legend_orange_dock_stations' : "🟠: Could find a dock but no guarantee",
        'colour_plot_legend_orange_bike_stations' : "🟠: Could find a bike but no guarantee",
        'colour_plot_legend_red_dock_stations' : "🔴: Will be hard to find a dock",
        'colour_plot_legend_red_bike_stations' : "🔴: Will be hard to find a bike",
        'colour_plot_legend_grey_stations' : "⚪: No data found for this time",
        'ditribution_plot_stations' : "Check Hourly Ride Distribution",
        'more_stations_nearby_stations' : '**Find more stations nearby and check their availabilities!**',
        'title_bixi_wrap' : "Bixi Wrap 2024",
        'intro_bixi_wrap' : 'Explore the past year through data',
        'comment_bixi_wrap' : 'Thanks to riders like you, Bixi has been used more than 13 Million times in the past year',
        'filtering_neighbourhood_bixi_wrap' : 'Filter by neighbourhood:',
        'most_frequented_stations_arrival_bixi_wrap' : "**Most Frequented Stations to Drop off Bikes:**",
        'most_frequented_stations_departure_bixi_wrap' : "**Most Frequented Stations to Pickup Bikes:**"
    },
    'FR' : {
        'tab_home' : 'Accueil',
        'tab_stations' : 'Exploration des stations',
        'tab_bixi_wrap' : 'Rétrospective Bixi',
        'all' : 'Tous',
        'departure' : 'départs',
        'arrival' : 'arrivées',
        'title_homepage' : "L'envers du décors des trajets Bixi",
        'intro_homepage' : "Découvrez à quel moment chaque station Bixi se remplit ou se vide selon le jour. Idéal pour planifier vos trajets ou comprendre le rythme cycliste de la ville.",
        'intro_stations' : "Découvrez le rythme de votre station et planifiez vos déplacements plus intelligemment",
        'station_selection_stations' : "Choisissez une station et commencez votre exploration",
        'day_selection_stations' : "Choisissez un jour de la semaine:",
        'list_of_day_selections_stations' : ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],
        'bike_dock_selection_stations' : "Vous cherchez un vélo ou un ancrage",
        'station_not_found_recently' : "Cette station n'a pas été utilisée depuis longtemps. Il est possible qu'elle a été enlevée ou déplacée autre part.",
        'station_stats_departure_stations' : "La plupart des départs ont lieu à :",
        'station_stats_arrival_stations' : "La plupart des arrivées ont lieu à :",
        'colour_plot_legend_green_dock_stations' : "🟢 : Bonnes chances de trouver un ancrage",
        'colour_plot_legend_green_bike_stations' : "🟢 : Bonnes chances de trouver un vélo",
        'colour_plot_legend_orange_dock_stations' : "🟠 : Possibilité de trouver un ancrage, sans garantie",
        'colour_plot_legend_orange_bike_stations' : "🟠 : Possibilité de trouver un vélo, sans garantie",
        'colour_plot_legend_red_dock_stations' : "🔴 : Difficile de trouver un ancrage",
        'colour_plot_legend_red_bike_stations' : "🔴 : Difficile de trouver un vélo",
        'colour_plot_legend_grey_stations' : "⚪: Aucune données trouvées",
        'ditribution_plot_stations' : "Consultez la distribution horaire des trajets",
        'more_stations_nearby_stations' : '**Trouvez d’autres stations à proximité et vérifiez leur disponibilité !**',
        'title_bixi_wrap' : "Rétrospective Bixi 2024",
        'intro_bixi_wrap' : "Explorez l’année écoulée à travers les données",
        'comment_bixi_wrap' : "Grâce à des cyclistes comme vous, Bixi a été utilisé plus de 13 millions de fois cette année",
        'filtering_neighbourhood_bixi_wrap' : 'Filtrer par quartier :',
        'most_frequented_stations_arrival_bixi_wrap' : "**Stations les plus fréquentées pour déposer un vélo :**",
        'most_frequented_stations_departure_bixi_wrap' : "**Stations les plus fréquentées pour prendre un vélo :**",
    }
}

T = texts[st.session_state.lang]

if "tab" not in st.session_state:
    st.session_state.tab = 'tab_home'


# ======================================
# Reading Data
# ======================================
station_stats = pd.read_csv('./bixi_station_stats.csv')
station_time = pd.read_csv('./bixi_time.csv')
recent_station = pd.read_csv('./recent_stations.csv')

day_of_week_dict = {
    'EN': {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    },
    'FR' : {
        0: 'Lundi',
        1: 'Mardi',
        2: 'Mercredi',
        3: 'Jeudi',
        4: 'Vendredi',
        5: 'Samedi',
        6: 'Dimanche'
    }
}

config = {
    'staticPlot': True,        # makes it completely static (no zoom, pan, hover)
    'displayModeBar': False,   # hides toolbar (download, zoom buttons)
    'responsive': True
}

# ======================================
# 🧭 HEADER
# ======================================
home_page, stations, bixi_wrap = st.tabs([T['tab_home'],T['tab_stations'], T['tab_bixi_wrap']], default=T['tab_home'])

with home_page:
    st.session_state.tab = 'tab_home'
    st.title(T['title_homepage'])
    st.markdown(T['intro_homepage'])

    # Load the image from your local folder
    img = Image.open("img_homepage2.png")
    st.image(img)


##########################################################################
with stations:
    st.session_state.tab = 'tab_stations'
    st.subheader(T['intro_stations'])

    station_selection_col, day_selection_col, bike_dock_selection_col = st.columns(spec=[0.40, 0.30, 0.30], gap='large')
    station_selection = station_selection_col.selectbox(
        T['station_selection_stations'],
        sorted(station_stats['station'].tolist()),
    )

    day_selection = day_selection_col.selectbox(
        T['day_selection_stations'],
        T['list_of_day_selections_stations'],
    )
    # day_numeric = day_of_week_dict.keys()[day_of_week_dict.values().index(day_selection)]
    day_numeric = next((k for k, v in day_of_week_dict[st.session_state.lang].items() if v == day_selection), None)

    bike_dock_selection = bike_dock_selection_col.segmented_control(
        T['bike_dock_selection_stations'], ['🚲', '🅿️'], selection_mode="single", default='🚲'
    )

    station_latitude = station_stats[station_stats['station']==station_selection].iloc[0]['latitude']
    station_longitude = station_stats[station_stats['station']==station_selection].iloc[0]['longitude']

    num_of_weeks_observed = len(station_time[station_time['weekday']==day_numeric][station_time[station_time['weekday']==day_numeric]['station']==station_selection])
    ex_station = station_time[station_time['station'] == station_selection]
    ex_station = ex_station[ex_station['weekday'] == day_numeric]
    ex_station['rides_departure'] = ex_station['rides_departure']/num_of_weeks_observed
    ex_station['rides_arrival'] = ex_station['rides_arrival']/num_of_weeks_observed
    ex_station['net_rides'] = ex_station['rides_departure'] - ex_station['rides_arrival'] if bike_dock_selection=='🅿️' else ex_station['rides_arrival'] - ex_station['rides_departure']

    busiest_hour_departure = ex_station[ex_station['rides_departure']==ex_station['rides_departure'].max()].iloc[0]['hour']
    busiest_hour_arrival = ex_station[ex_station['rides_arrival']==ex_station['rides_arrival'].max()].iloc[0]['hour']

    ex_station = ex_station.sort_values("hour")

    st.divider()

    if station_selection not in recent_station['station'].tolist():
        st.warning(body=T['station_not_found_recently'], icon='⚠️')

    # Resume Insights
    departure_stats_col, arrival_stats_col, colour_legend_col = st.columns(spec=[0.33, 0.33, 0.34], gap='small')
    departure_stats_col.metric(T['station_stats_departure_stations'], f"{busiest_hour_departure}h", f"approx {int(ex_station['rides_departure'].max())} {T['departure']}") 
    arrival_stats_col.metric(T['station_stats_arrival_stations'], f"{busiest_hour_arrival}h", f"approx {int(ex_station['rides_arrival'].max())} {T['arrival']}")
    with colour_legend_col.container(border=True):
        if bike_dock_selection=='🅿️':
            st.markdown(T['colour_plot_legend_green_dock_stations'])
            st.markdown(T['colour_plot_legend_orange_dock_stations'])
            st.markdown(T['colour_plot_legend_red_dock_stations'])
        else:
            st.markdown(T['colour_plot_legend_green_bike_stations'])
            st.markdown(T['colour_plot_legend_orange_bike_stations'])
            st.markdown(T['colour_plot_legend_red_bike_stations'])
        st.markdown(T['colour_plot_legend_grey_stations'])

    # Colour dot plot
    colour_dot_plot = generate_colour_dot_plot(df=station_time, station_name=station_selection, weekday=day_numeric, bike_or_dock=bike_dock_selection)

    # Display in Streamlit
    st.plotly_chart(colour_dot_plot, use_container_width=False, config=config, key='main')



    # Distribution Plot -----------------------------------------
    distribution_plot = generate_dist_plot(station_time, station_selection, day_numeric, bike_dock_selection, st.session_state.lang)

    with st.expander(T['ditribution_plot_stations']):
        st.plotly_chart(distribution_plot, use_container_width=False, config=config, key='dist_plot')

    # Finding Stations Nearby -------------------------------------
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        
        a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
        return int(round((2 * R * atan2(sqrt(a), sqrt(1 - a))), 0))

    def nearby_stations(df, lat, lon, radius_m=500):
        df = df.copy()
        df["distance_m"] = df.apply(
            lambda row: haversine(lat, lon, row["latitude"], row["longitude"]),
            axis=1
        )
        close_stations_df = df[df["distance_m"] <= radius_m].sort_values("distance_m")
        return close_stations_df[['station', 'distance_m']].iloc[1:, :]
    
    nearby_stations_df = nearby_stations(station_stats, station_latitude, station_longitude, radius_m=500).head(5)
    
    with st.container(border=True):
        st.markdown(T['more_stations_nearby_stations'])
        for nearby_station in range(len(nearby_stations_df)):
            station_selection = nearby_stations_df.iloc[nearby_station]['station']
            with st.expander(f"{station_selection}, {nearby_stations_df.iloc[nearby_station]['distance_m']} m ⚠️" if station_selection not in recent_station['station'].tolist() else f"{station_selection}, {nearby_stations_df.iloc[nearby_station]['distance_m']} m"):
                colour_dot_plot = generate_colour_dot_plot(df=station_time, station_name=station_selection, weekday=day_numeric, bike_or_dock=bike_dock_selection)
                # Display in Streamlit
                st.plotly_chart(colour_dot_plot, use_container_width=False, config=config, key=f"colour_plot_{station_selection}")
                if station_selection not in recent_station['station'].tolist():
                    st.warning(body=T['station_not_found_recently'], icon='⚠️')



#########################################
with bixi_wrap:
    st.session_state.tab = 'tab_bixi_wrap'
    st.header(T['title_bixi_wrap'])
    st.subheader(T['intro_bixi_wrap'])

    bixi_wrap_text_col, bixi_wrap_bike_dock_selection_col = st.columns(spec=[0.6, 0.4])

    bixi_wrap_text_col.write(T['comment_bixi_wrap'])
    bixi_wrap_bike_dock_selection = bixi_wrap_bike_dock_selection_col.segmented_control(
        "", ['🚲', '🅿️'], selection_mode="single", default='🚲'
    )

    map_col, stats_col = st.columns(spec=[0.7, 0.3], gap='small')

    with stats_col.container(border=True):
        arrondissement_selection = st.selectbox(T['filtering_neighbourhood_bixi_wrap'], [T['all']] + station_stats['arrondissement'].unique().tolist())
        rank_station = station_stats['rides_departure'].nlargest(3) if arrondissement_selection==T['all'] else station_stats[station_stats['arrondissement']==arrondissement_selection]['rides_departure'].nlargest(3)
        rank_station_list = [f"{station_stats['station'].loc[rank_station.index[0]]}", f"{station_stats['station'].loc[rank_station.index[1]]}", f"{station_stats['station'].loc[rank_station.index[2]]}"]
        rank_station_list_lon_lat = {
            'lon' : [station_stats['longitude'].loc[rank_station.index[0]], station_stats['longitude'].loc[rank_station.index[1]], station_stats['longitude'].loc[rank_station.index[2]]],
            'lat' : [station_stats['latitude'].loc[rank_station.index[0]], station_stats['latitude'].loc[rank_station.index[1]], station_stats['latitude'].loc[rank_station.index[2]]]
        }
        
        if bixi_wrap_bike_dock_selection=='🅿️':
            st.markdown(T['most_frequented_stations_arrival_bixi_wrap'])
            st.markdown(f"🥇 {rank_station_list[0]} : {int(station_stats['rides_arrival'].loc[rank_station.index[0]])} {T['arrival']}")
            st.markdown(f"🥈 {rank_station_list[1]} : {int(station_stats['rides_arrival'].loc[rank_station.index[1]])} {T['arrival']}")
            st.markdown(f"🥉 {rank_station_list[2]} : {int(station_stats['rides_arrival'].loc[rank_station.index[2]])} {T['arrival']}")
        else:
            st.markdown(T['most_frequented_stations_departure_bixi_wrap'])
            st.markdown(f"🥇 {rank_station_list[0]} : {int(station_stats['rides_departure'].loc[rank_station.index[0]])} {T['departure']}") 
            st.markdown(f"🥈 {rank_station_list[1]} : {int(station_stats['rides_departure'].loc[rank_station.index[1]])} {T['departure']}")
            st.markdown(f"🥉 {rank_station_list[2]} : {int(station_stats['rides_departure'].loc[rank_station.index[2]])} {T['departure']}")

    # Define PyDeck layer
    def get_map_attributes(row):
        if (row['station'] in rank_station_list) and (row['arrondissement'] == arrondissement_selection):
            color = [181, 126, 220, 100]
        elif (row['station'] in rank_station_list) and (arrondissement_selection==T['all']):
            color = [181, 126, 220, 100]
        elif arrondissement_selection==T['all']:
            color = [255, 0, 0, 100]
        elif row['arrondissement'] == arrondissement_selection:
            color = [255, 0, 0, 100]
        else:
            color = [255, 0, 0, 20]
        return color
  
    map_df = station_stats.copy()
    map_df["color"] = map_df.apply(get_map_attributes, axis=1)

    layer = pdk.Layer(
        "ColumnLayer",                # 3D column markers
        data=map_df,
        get_position=["longitude", "latitude"],
        get_elevation="rides_arrival" if bixi_wrap_bike_dock_selection=='🅿️' else 'rides_departure',   # height of the column
        elevation_scale=0.05,           # scaling factor for height
        radius=40,                    # radius of the column
        get_fill_color="color",
        pickable=True,
        auto_highlight=True
    )

    # Define view
    view_state = pdk.ViewState(
        longitude=rank_station_list_lon_lat['lon'][0],
        latitude=rank_station_list_lon_lat['lat'][0],
        zoom=11,
        pitch=35  # tilt the map for 3D effect
    )

    # Render map in Streamlit
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="dark", tooltip={"text": "{station} : {rides_arrival}"} if bixi_wrap_bike_dock_selection=='🅿️' else {"text": "{station} : {rides_departure}"})
    map_col.pydeck_chart(r)
